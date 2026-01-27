import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Configuration
MODEL_PATH = 'models/model.h5'
CLASS_INDICES_PATH = 'models/class_indices.txt'

# Page Config
st.set_page_config(
    page_title="CropGuard AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Wow" Factor
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .main {
        background-color: transparent;
        color: #1a1a1a;
        font-family: 'Inter', 'Segoe UI', serif;
    }
    
    /* Header Section */
    .header-container {
        background: rgba(255, 255, 255, 0.8);
        padding: 2.5rem;
        border-radius: 20px;
        backdrop-filter: blur(10px);
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
        text-align: center;
    }
    
    h1 {
        color: #1b5e20;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #4a4a4a;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Instructions Card */
    .instruction-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #2e7d32;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    /* Input/Result Headings */
    .section-header {
        color: #2e7d32;
        font-weight: 700;
        border-bottom: 2px solid #81c784;
        padding-bottom: 8px;
        margin-bottom: 15px;
        font-size: 1.3rem;
    }
    
    /* Scan Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(to right, #43a047, #1b5e20);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(67, 160, 71, 0.3);
    }
    
    /* Result Card */
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
        text-align: center;
        margin-top: 1rem;
    }
    
    .prediction-text {
        color: #1b5e20;
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .confidence-high {
        color: #2e7d32;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    .confidence-low {
        color: #c62828;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    /* Treatment Section */
    .treatment-section {
        background-color: #f1f8e9;
        padding: 1.5rem;
        border-radius: 15px;
        margin-top: 1.5rem;
        border-left: 5px solid #689f38;
    }
    
    .treatment-header {
        color: #33691e;
        font-weight: 700;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
</style>
""", unsafe_allow_html=True)

# Treatment Dictionary (case must match dataset folder names exactly)
TREATMENTS = {
    "Tomato_Early-blight": "Apply copper-based fungicides. Prune affected leaves to improve air circulation. Ensure proper spacing between plants.",
    "Tomato_late-blight": "Remove and destroy infected plants immediately. Apply fungicides like chlorothalonil. Avoid overhead watering to reduce leaf wetness.", 
    "Tomato_bac-spot": "Use certified disease-free seeds. Apply copper sprays early. Remove infected plant debris. Practice crop rotation with non-solanaceous crops.",
    "tomato_leaf-mold": "Improve air circulation by pruning and spacing. Reduce greenhouse humidity. Apply fungicides if the infection is severe.",
    "tomato_mosaic_virus": "There is no cure. Remove and destroy infected plants to prevent spread. Control aphids and sanitize tools. Plant resistant varieties.",
    "Tomato_Healthy": "No treatment needed. Continue good agricultural practices: consistent watering, fertilization, and regular monitoring."
}

# Helper Function to Load Model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# App Header
st.markdown("""
<div class="header-container">
    <h1>🌿 CropGuard AI</h1>
    <p class="subtitle">Intelligent Plant Health Diagnostic Tool</p>
</div>
""", unsafe_allow_html=True)

# Instructions Section
with st.expander("ℹ️ How to use CropGuard AI", expanded=True):
    st.markdown("""
    <div class="instruction-card">
        <strong>Simple 3-Step Process:</strong>
        <ol>
            <li><strong>Input:</strong> Upload a clear photo of a single leaf or use your camera.</li>
            <li><strong>Analyze:</strong> Click the 'Analyze Leaf' button to run the AI diagnostic.</li>
            <li><strong>Results:</strong> Review the prediction, confidence level, and treatment advice.</li>
        </ol>
        <em>Tip: For best results, ensure the leaf is well-lit and centered in the frame.</em>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("About")
st.sidebar.info(
    "CropGuard AI leverages Cutting-edge Computer Vision (MobileNetV2) "
    "to help farmers and gardeners identify crop diseases instantly. "
    "Early detection is key to preventing crop loss."
)
st.sidebar.markdown("---")
st.sidebar.markdown("### Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.65, 0.05)

# Main Interface
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<p class="section-header">📸 Image Input</p>', unsafe_allow_html=True)
    # Image Source Selection
    input_method = st.radio("Select Input Method:", ("Upload Image", "Use Camera"), horizontal=True)
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])
        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
    else:
        camera_image = st.camera_input("Take a picture")
        image = None
        if camera_image is not None:
            image = Image.open(camera_image)
            # st.image(image, caption="Captured Image", use_column_width=True) # Camera input shows preview already

with col2:
    st.markdown('<p class="section-header">🔍 Analysis Results</p>', unsafe_allow_html=True)
    
    if image is not None:
        if st.button("Analyze Leaf", type="primary"):
            with st.spinner("Analyzing image..."):
                # Load model
                model = load_model()
                
                if model is not None:
                    # Preprocess
                    img = image.resize((224, 224))
                    img_array = np.array(img)
                    if img_array.shape[-1] == 4: # Handle PNG with alpha channel
                        img_array = img_array[..., :3]
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = img_array / 255.0  # Normalize
                    
                    # Predict
                    predictions = model.predict(img_array)
                    confidence = np.max(predictions)
                    class_idx = np.argmax(predictions)
                    
                    # Load class names
                    class_names = []
                    if os.path.exists(CLASS_INDICES_PATH):
                        with open(CLASS_INDICES_PATH, 'r') as f:
                            indices = eval(f.read()) # Safe for trusted local file
                            idx_to_class = {v: k for k, v in indices.items()}
                            class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
                    else:
                        try:
                            if os.path.exists('dataset'):
                                subdirs = [d for d in os.listdir('dataset') if os.path.isdir(os.path.join('dataset', d))]
                                if subdirs:
                                    target_dir = os.path.join('dataset', subdirs[0])
                                    class_names = sorted([d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))])
                        except Exception as e:
                            st.error(f"Error determining classes: {e}")

                    if not class_names:
                        st.error("Could not determine class labels. Please ensure model is trained.")
                        predicted_class = "Unknown"
                        predicted_class_display = "Unknown"
                    elif class_idx < len(class_names):
                        predicted_class = class_names[class_idx]
                        if confidence < confidence_threshold:
                            predicted_class_display = "Uncertain"
                        else:
                            predicted_class_display = predicted_class.replace('_', ' ').title()
                    else:
                        predicted_class = "Unknown"
                        predicted_class_display = "Unknown"

                    # Display
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="prediction-text">{predicted_class_display}</div>
                        <div class="{ 'confidence-high' if confidence > confidence_threshold else 'confidence-low'}">
                            Confidence: {confidence * 100:.2f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Handing Low Confidence
                    if confidence < confidence_threshold:
                        st.warning(f"""
                            **Low Confidence Warning** ({confidence*100:.1f}%)
                            The model is uncertain. The image might be unclear or belong to an unsupported type.
                            Please consult an agricultural expert.
                        """)
                    
                    # Treatment Recommendation
                    treatment = TREATMENTS.get(predicted_class, "Consult an agricultural expert for specific advice on this condition.")
                    
                    st.markdown(f"""
                    <div class="treatment-section">
                        <div class="treatment-header">💊 Recommended Treatment</div>
                        <p>{treatment}</p>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.error("Model not found! Please train the model first by running `train.py`.")
    else:
        st.info("Please upload an image or use the camera to start analysis.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #444; margin-bottom: 20px;'>© 2026 CropGuard AI • Hackathon Prototype</div>", unsafe_allow_html=True)
