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
        background-color: #ffffff;
    }
    
    .main {
        background-color: transparent;
        color: #1a1a1a !important;
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    }
    
    /* Header Section */
    .header-container {
        padding: 2rem 0;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    h1 {
        color: #1b5e20 !important;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .subtitle {
        color: #555555 !important;
        font-size: 1.5rem;
        font-weight: 400;
    }
    
    /* Instructions */
    .instruction-text {
        color: #333333 !important;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    /* Section Headers */
    .section-header {
        color: #1b5e20 !important;
        font-weight: 700;
        font-size: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background-color: #2e7d32 !important;
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #1b5e20 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Result Card - The Focal Point */
    .result-card {
        background: #f8f9fa;
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 1rem;
        border: 1px solid #e0e0e0;
    }
    
    .prediction-text {
        color: #000000 !important;
        font-size: 3.5rem;
        font-weight: 900;
        line-height: 1.2;
        margin-bottom: 1rem;
        letter-spacing: -1px;
    }
    
    .confidence-tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    
    .confidence-high {
        background-color: #e8f5e9;
        color: #1b5e20;
    }
    
    .confidence-low {
        background-color: #ffebee;
        color: #c62828;
    }
    
    /* Treatment Section */
    .treatment-section {
        margin-top: 2rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e0e0e0;
        text-align: left;
    }
    
    .treatment-header {
        color: #333333 !important;
        font-weight: 700;
        font-size: 1.3rem;
        margin-bottom: 0.75rem;
    }

    .treatment-section p {
        color: #444444 !important;
        font-size: 1.15rem;
        line-height: 1.6;
    }

    /* Sidebar adjustments */
    [data-testid="stSidebar"] {
        background-color: #f5f5f5;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Widget Labels & Input Text */
    /* Target Radio button options ("Upload Image", "Use Camera") */
    div[role="radiogroup"] p {
        color: #1a1a1a !important;
        font-weight: 600;
        font-size: 1.05rem;
    }
    
    /* Target Widget Labels ("Upload an image", "Input Method:") */
    label[data-testid="stWidgetLabel"] p {
        color: #1a1a1a !important;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Alerts (Success, Info, Warning, Error) Text */
    div[data-baseweb="alert"] div {
        color: #1a1a1a !important;
        font-weight: 500;
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
    <h1>CropGuard AI</h1>
    <p class="subtitle">Instant Plant Disease Diagnosis</p>
</div>
""", unsafe_allow_html=True)

# Instructions Section
with st.container():
    st.markdown("""
    <div class="instruction-text">
        <strong>How it works:</strong> Upload a clear leaf photo → Get instant diagnosis & treatment advice.
    </div>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("About")
st.sidebar.info(
    "CropGuard AI helps farmers identify crop diseases instantly using computer vision."
)
st.sidebar.markdown("---")
st.sidebar.markdown("### Settings")
confidence_threshold = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.65, 0.05)

# Main Interface
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<p class=\"section-header\">Add a Leaf Photo</p>', unsafe_allow_html=True)
    # Image Source Selection
    input_method = st.radio("Input Method:", ("Upload Image", "Use Camera"), horizontal=True)
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
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
    st.markdown('<p class=\"section-header\">Analysis Results</p>', unsafe_allow_html=True)
    
    if image is not None:
        if st.button("Analyze Leaf", type="primary"):
            with st.spinner("Analyzing..."):
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
                        <div class="confidence-tag { 'confidence-high' if confidence > confidence_threshold else 'confidence-low'}">
                            Confidence: {confidence * 100:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Handing Low Confidence
                    if confidence < confidence_threshold:
                        st.warning(f"""
                            **Note:** Confidence is low ({confidence*100:.1f}%). 
                            The model isn't sure about this result. It might not be a leaf or the image is blurry.
                        """)
                    
                    # Treatment Recommendation
                    treatment = TREATMENTS.get(predicted_class, "Please consult an agricultural expert for advice.")
                    
                    st.markdown(f"""
                    <div class="treatment-section">
                        <div class="treatment-header">Recommended Action</div>
                        <p>{treatment}</p>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.error("Model not found! Please train the model first by running `train.py`.")
    else:
        st.info("Please upload an image or use the camera to start analysis.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666; font-size: 0.9rem;'>© 2026 CropGuard AI • Hackathon Prototype</div>", unsafe_allow_html=True)
