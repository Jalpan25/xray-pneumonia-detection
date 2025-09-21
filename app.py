import os
from pathlib import Path
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import pandas as pd

# ==========================
# 🎨 Custom CSS Styling
# ==========================
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --accent-color: #F18F01;
        --success-color: #06D6A0;
        --warning-color: #FFD23F;
        --error-color: #F71735;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(90deg, #2E86AB, #A23B72);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Enhanced metrics styling */
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid var(--primary-color);
        margin: 0.5rem 0;
    }
    
    /* Model card styling */
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .model-card:hover {
        transform: translateY(-5px);
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        margin: 0.25rem;
    }
    
    .status-pneumonia {
        background-color: #ffe6e6;
        color: #d32f2f;
        border: 2px solid #ffcdd2;
    }
    
    .status-normal {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #c8e6c9;
    }
    
    /* Progress bars enhancement */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--success-color), var(--primary-color));
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0px 0px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color);
        color: white;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Upload area styling */
    .uploadedFile {
        border: 2px dashed var(--primary-color);
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9ff;
    }
    
    /* Custom alert boxes */
    .custom-alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        color: #155724;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .custom-alert-info {
        background-color: #cce7ff;
        border: 1px solid #99d6ff;
        border-radius: 8px;
        color: #004085;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Enhanced dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==========================
# 🔹 Fix for "Unknown layer: Cast"
# ==========================
class CastLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

# ==========================
# 🎨 Page Configuration
# ==========================
st.set_page_config(
    page_title="AI Pneumonia Detection",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🫁"
)

# ==========================
# 🎨 Custom Header
# ==========================
st.markdown("""
<div class="main-header">
    <h1>🫁 AI-Powered Pneumonia Detection</h1>
    <p>Advanced Multi-Model Comparison System for Chest X-Ray Analysis</p>
</div>
""", unsafe_allow_html=True)

# ==========================
# 📋 Model Configuration
# ==========================
MODEL_FILES = {
    "DenseNet121": {
        "file": "best_densenet121.h5", 
        "color": "#FF6B6B", 
        "description": "Dense connectivity between layers for feature reuse",
        "icon": "🔗"
    },
    "MobileNetV2": {
        "file": "best_mobilenetv2.h5", 
        "color": "#4ECDC4", 
        "description": "Lightweight mobile-optimized architecture",
        "icon": "📱"
    },
    "ResNet50": {
        "file": "best_resnet50.h5", 
        "color": "#45B7D1", 
        "description": "Residual connections for deeper networks",
        "icon": "🏗️"
    },
}

IMG_SIZE = (224, 224)

# ==========================
# 🔧 Utility Functions
# ==========================
def create_model_card(name, info, loaded=True):
    """Create a styled model card"""
    status_icon = "✅" if loaded else "❌"
    status_text = "Loaded" if loaded else "Failed to Load"
    
    return f"""
    <div class="model-card">
        <h3>{info['icon']} {name}</h3>
        <p>{info['description']}</p>
        <div style="margin-top: 1rem;">
            <span style="background: rgba(255,255,255,0.2); padding: 0.25rem 0.75rem; border-radius: 15px;">
                {status_icon} {status_text}
            </span>
        </div>
    </div>
    """

def create_prediction_badge(prediction, probability, threshold):
    """Create a styled prediction badge"""
    if prediction == "PNEUMONIA":
        return f'<div class="status-indicator status-pneumonia">🔴 PNEUMONIA ({probability:.1%})</div>'
    else:
        return f'<div class="status-indicator status-normal">🟢 NORMAL ({1-probability:.1%})</div>'

# ==========================
# 🔄 Load Models
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
                model_info[name] = {"loaded": True, **config}
            except Exception as e:
                st.warning(f"⚠️ Failed to load {name}: {str(e)}")
                model_info[name] = {"loaded": False, "error": str(e), **config}
        else:
            model_info[name] = {"loaded": False, "error": "File not found", **config}
    
    return models, model_info

def preprocess_image(pil_img: Image.Image):
    """Preprocess image for model prediction"""
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def calculate_ensemble(results, method="average"):
    """Calculate ensemble prediction from multiple models"""
    if not results:
        return None, None
    
    probs = [r["raw_prob"] for r in results]
    
    if method == "average":
        ensemble_prob = np.mean(probs)
    elif method == "weighted":
        weights = [0.35, 0.35, 0.3][:len(probs)]
        ensemble_prob = np.average(probs, weights=weights)
    elif method == "voting":
        votes = [1 if p >= 0.5 else 0 for p in probs]
        ensemble_prob = np.mean(votes)
    
    return ensemble_prob, "PNEUMONIA" if ensemble_prob >= 0.5 else "NORMAL"

# ==========================
# 🎛️ Sidebar Configuration
# ==========================
with st.sidebar:
    st.markdown("### 🎛️ Configuration Panel")
    
    threshold = st.slider(
        "🎯 Decision Threshold", 
        0.1, 0.9, 0.5, 0.01,
        help="Probability threshold for pneumonia classification"
    )
    
    ensemble_method = st.selectbox(
        "⚖️ Ensemble Method", 
        ["average", "weighted", "voting"],
        help="Method for combining model predictions"
    )
    
    show_details = st.checkbox("📊 Show Model Details", True)
    
    st.markdown("---")
    st.markdown("### 📈 Model Performance")

# ==========================
# 🏗️ Load Models
# ==========================
try:
    models, model_info = load_all_models()
    loaded_models = {k: v for k, v in models.items() if model_info[k]["loaded"]}
    
    if not loaded_models:
        st.error("❌ No models could be loaded. Please check your model files.")
        st.stop()
    
    with st.sidebar:
        st.success(f"🎯 {len(loaded_models)}/{len(MODEL_FILES)} models loaded")
    
except Exception as e:
    st.error(f"❌ Could not load models: {e}")
    st.stop()

# ==========================
# 📋 Model Information Display
# ==========================
if show_details:
    st.markdown("## 🏛️ Model Architecture Overview")
    
    # Model cards in columns
    cols = st.columns(len(MODEL_FILES))
    for i, (name, info) in enumerate(MODEL_FILES.items()):
        with cols[i]:
            loaded = model_info[name]["loaded"]
            st.markdown(create_model_card(name, info, loaded), unsafe_allow_html=True)
    
    # Architecture comparison table
    st.markdown("### 📊 Technical Specifications")
    arch_data = pd.DataFrame({
        "Model": ["DenseNet121", "MobileNetV2", "ResNet50"],
        "Parameters (M)": [8.1, 3.4, 25.6],
        "Depth (Layers)": [121, 154, 50],
        "Primary Use": ["Feature Reuse", "Mobile Deployment", "Deep Learning"],
        "Year Introduced": [2017, 2018, 2015]
    })
    
    st.dataframe(
        arch_data, 
        use_container_width=True,
        column_config={
            "Parameters (M)": st.column_config.NumberColumn("Parameters (M)", format="%.1f"),
            "Depth (Layers)": st.column_config.NumberColumn("Depth", format="%d"),
        }
    )

# ==========================
# 📤 Main Upload Interface
# ==========================
st.markdown("## 📤 Upload Chest X-Ray")

uploaded = st.file_uploader(
    "Choose a chest X-ray image for analysis", 
    type=["jpg", "jpeg", "png", "bmp"],
    help="Upload a clear chest X-ray image. Supported formats: JPG, JPEG, PNG, BMP"
)

if uploaded is not None:
    # Create two columns for image and info
    col1, col2 = st.columns([1, 2])
    
    with col1:
        img = Image.open(uploaded)
        st.image(img, caption="📸 Uploaded X-ray", use_column_width=True)
        
        # Image info card
        st.markdown(f"""
        <div class="custom-alert-info">
            <strong>📏 Image Information:</strong><br>
            • Size: {img.size[0]} × {img.size[1]} pixels<br>
            • Format: {img.format}<br>
            • Mode: {img.mode}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🔍 Analysis in Progress")
        
        # Processing
        x = preprocess_image(img)
        results = []
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (name, model) in enumerate(loaded_models.items()):
            status_text.text(f"🤖 Analyzing with {name}...")
            
            preds = model.predict(x, verbose=0)
            prob = float(preds[0][0])
            label = "PNEUMONIA" if prob >= threshold else "NORMAL"
            confidence = prob if label == "PNEUMONIA" else 1 - prob
            
            results.append({
                "model": name,
                "prediction": label,
                "confidence": confidence,
                "raw_prob": prob,
                "color": model_info[name]["color"],
                "icon": model_info[name]["icon"]
            })
            
            progress_bar.progress((i + 1) / len(loaded_models))
        
        status_text.text("✅ Analysis Complete!")
        progress_bar.empty()
    
    # ==========================
    # 📊 Results Dashboard
    # ==========================
    st.markdown("## 📊 Analysis Results Dashboard")
    
    # Quick overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    pneumonia_count = sum(1 for r in results if r["prediction"] == "PNEUMONIA")
    normal_count = len(results) - pneumonia_count
    avg_confidence = np.mean([r["confidence"] for r in results])
    max_prob = max([r["raw_prob"] for r in results])
    
    with col1:
        st.metric("🤖 Models Analyzed", len(results))
    with col2:
        st.metric("🔴 Pneumonia Predictions", pneumonia_count)
    with col3:
        st.metric("🟢 Normal Predictions", normal_count)
    with col4:
        st.metric("📈 Highest Confidence", f"{max_prob:.1%}")
    
    # Tabbed results interface
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Summary Report", 
        "📈 Visual Analysis", 
        "🎯 Ensemble Results", 
        "🔬 Detailed Breakdown"
    ])
    
    with tab1:
        st.markdown("### 📋 Model Predictions Summary")
        
        # Enhanced results table
        df_results = pd.DataFrame([
            {
                "🤖 Model": f"{r['icon']} {r['model']}",
                "🎯 Prediction": r["prediction"],
                "📊 Confidence": f"{r['confidence']:.1%}",
                "🔢 Raw Probability": f"{r['raw_prob']:.4f}",
                "📈 Status": "🔴 PNEUMONIA" if r["prediction"] == "PNEUMONIA" else "🟢 NORMAL"
            }
            for r in results
        ])
        
        st.dataframe(df_results, use_container_width=True, hide_index=True)
        
        # Consensus analysis
        st.markdown("### 🤝 Model Consensus")
        unanimous = len(set(r["prediction"] for r in results)) == 1
        majority_pred = max(set(r["prediction"] for r in results), 
                          key=lambda x: sum(1 for r in results if r["prediction"] == x))
        
        if unanimous:
            st.markdown(
                '<div class="custom-alert-success"><strong>✅ Unanimous Agreement:</strong> '
                f'All models predict <strong>{majority_pred}</strong></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="custom-alert-info"><strong>⚖️ Mixed Results:</strong> '
                f'Majority prediction is <strong>{majority_pred}</strong></div>',
                unsafe_allow_html=True
            )
    
    with tab2:
        st.markdown("### 📊 Probability Visualization")
        
        # Probability comparison chart
        prob_data = pd.DataFrame({
            'Model': [f"{r['icon']} {r['model']}" for r in results],
            'Pneumonia_Probability': [r["raw_prob"] for r in results]
        })
        
        st.bar_chart(prob_data.set_index('Model'), height=400)
        
        # Threshold indicator
        st.markdown(f"""
        <div class="custom-alert-info">
            🎯 <strong>Decision Threshold:</strong> {threshold:.2f} 
            (Values above this threshold indicate pneumonia)
        </div>
        """, unsafe_allow_html=True)
        
        # Individual model metrics
        st.markdown("### 🎯 Individual Model Performance")
        cols = st.columns(len(results))
        
        for i, r in enumerate(results):
            with cols[i]:
                delta_color = "normal" if r['raw_prob'] >= threshold else "inverse"
                st.metric(
                    f"{r['icon']} {r['model']}", 
                    f"{r['raw_prob']:.3f}",
                    delta=f"{r['raw_prob'] - threshold:+.3f}",
                    delta_color=delta_color
                )
    
    with tab3:
        st.markdown("### 🎯 Ensemble Prediction Analysis")
        
        ensemble_prob, ensemble_pred = calculate_ensemble(results, ensemble_method)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Ensemble result card
            st.markdown(f"""
            <div class="model-card">
                <h3>🎯 Final Ensemble Result</h3>
                <h2 style="color: {'#ff4444' if ensemble_pred == 'PNEUMONIA' else '#44ff44'};">
                    {ensemble_pred}
                </h2>
                <p><strong>Probability:</strong> {ensemble_prob:.1%}</p>
                <p><strong>Method:</strong> {ensemble_method.title()}</p>
                <p><strong>Confidence:</strong> {abs(ensemble_prob - 0.5) * 2:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 📊 Ensemble vs Individual Comparison")
            
            comparison_data = pd.DataFrame({
                'Source': ["🎯 Ensemble"] + [f"{r['icon']} {r['model']}" for r in results],
                'Probability': [ensemble_prob] + [r["raw_prob"] for r in results]
            })
            
            st.bar_chart(comparison_data.set_index('Source'), height=300)
        
        # Method explanation
        method_explanations = {
            "average": "Simple average of all model probabilities",
            "weighted": "Weighted average favoring better-performing models",
            "voting": "Majority voting based on binary predictions"
        }
        
        st.info(f"📝 **{ensemble_method.title()} Method:** {method_explanations[ensemble_method]}")
    
    with tab4:
        st.markdown("### 🔬 Detailed Model Analysis")
        
        for r in results:
            with st.expander(f"{r['icon']} {r['model']} - {r['prediction']} (Click to expand)", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **🎯 Prediction:** {r['prediction']}  
                    **🔢 Raw Probability:** {r['raw_prob']:.4f}  
                    **📊 Confidence Level:** {r['confidence']:.2%}  
                    **🏗️ Architecture:** {model_info[r['model']]['description']}
                    """)
                    
                    # Threshold comparison
                    if r['raw_prob'] >= threshold:
                        st.error(f"🔴 Above threshold ({threshold:.2f}) - Indicates PNEUMONIA")
                    else:
                        st.success(f"🟢 Below threshold ({threshold:.2f}) - Indicates NORMAL")
                
                with col2:
                    st.markdown("**📊 Probability Breakdown:**")
                    
                    # Pneumonia probability
                    st.markdown(f"**Pneumonia Risk:** {r['raw_prob']:.1%}")
                    st.progress(r['raw_prob'])
                    
                    # Normal probability  
                    st.markdown(f"**Normal Likelihood:** {1-r['raw_prob']:.1%}")
                    st.progress(1-r['raw_prob'])
                    
                    # Confidence indicator
                    st.markdown(f"**Overall Confidence:** {r['confidence']:.1%}")
                    st.progress(r['confidence'])

else:
    # ==========================
    # 🏠 Welcome Screen
    # ==========================
    st.markdown("## 🏠 Welcome to AI Pneumonia Detection")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="custom-alert-info">
            <h4>🤖 Multi-Model Analysis</h4>
            <p>Compare predictions from three state-of-the-art CNN architectures for robust diagnosis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="custom-alert-info">
            <h4>🎯 Ensemble Learning</h4>
            <p>Combine multiple models using advanced ensemble methods for improved accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="custom-alert-info">
            <h4>📊 Detailed Analytics</h4>
            <p>Comprehensive analysis with confidence scores, visualizations, and interpretability.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature showcase
    st.markdown("### 🎯 System Capabilities")
    
    capabilities_data = pd.DataFrame({
        "🔧 Feature": [
            "🤖 Multi-Model Comparison",
            "🎯 Ensemble Predictions", 
            "📊 Confidence Analysis",
            "📈 Interactive Visualizations",
            "🔬 Detailed Metrics",
            "⚡ Real-time Processing"
        ],
        "📝 Description": [
            "Compare predictions from DenseNet121, MobileNetV2, and ResNet50",
            "Combine model predictions using average, weighted, or voting methods",
            "Analyze prediction confidence and model agreement levels",
            "Interactive charts and progress bars for better understanding",
            "Comprehensive metrics including probability scores and thresholds",
            "Fast inference with optimized model loading and caching"
        ]
    })
    
    st.dataframe(capabilities_data, use_container_width=True, hide_index=True)

# ==========================
# 🔒 Footer
# ==========================
st.markdown("---")
st.markdown("""
<div style='background: linear-gradient(90deg, #2E86AB, #A23B72); padding: 2rem; border-radius: 10px; text-align: center; color: white; margin-top: 3rem;'>
    <h4>⚠️ Medical Disclaimer</h4>
    <p style='margin: 0; opacity: 0.9;'>
        This application is for educational and research purposes only. It should not be used as a substitute 
        for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers 
        for medical decisions.
    </p>
</div>
""", unsafe_allow_html=True)