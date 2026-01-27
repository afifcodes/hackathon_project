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
    .main {
        background-color: #f8f9fa;
        color: #333333;
        font-family: 'Inter', sans-serif;
    }
    
    /* Title */
    h1 {
        color: #2E7D32;
        font-weight: 700;
        text-align: center;
        padding-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Subheader */
    h3 {
        color: #4CAF50;
        border-bottom: 2px solid #4CAF50;
        padding-bottom: 10px;
    }
    
    /* Scan Button */
    .stButton>button {
        background-image: linear-gradient(to right, #4CAF50, #2E7D32);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-size: 18px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    
    /* Result Card */
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        margin-top: 2rem;
        text-align: center;
        border-left: 5px solid #4CAF50;
    }
    
    .confidence-high {
        color: #2E7D32;
        font-weight: bold;
    }
    
    .confidence-low {
        color: #D32F2F;
        font-weight: bold;
    }
    
    /* Warning Box */
    .warning-box {
        background-color: #FFF3E0;
        border: 1px solid #FFB74D;
        color: #E65100;
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
        text-align: left;
    }
    
    /* Treatment Section */
    .treatment-section {
        background-color: #E8F5E9;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        border: 1px solid #C8E6C9;
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
st.title("🌿 CropGuard AI")
st.markdown("**Advanced Plant Disease Detection System**")
st.markdown("Upload a leaf image to identify diseases and get treatment recommendations.")

# Sidebar
st.sidebar.title("About")
st.sidebar.info(
    "This AI application uses a Transfer Learning model (MobileNetV2) "
    "to classify crop leaf diseases with high accuracy. "
    "Designed for rapid diagnosis in the field."
)
st.sidebar.markdown("---")
st.sidebar.markdown("### Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.65, 0.05)

# Main Interface
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Image")
    # Image Source Selection
    input_method = st.radio("Select Input Method:", ("Upload Image", "Use Camera"))
    
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
    st.subheader("Analysis Results")
    
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
                            # indices is {'class': 0, ...}
                            # Invert to {0: 'class', ...}
                            idx_to_class = {v: k for k, v in indices.items()}
                            class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
                    else:
                        try:
                            # Fallback: Keras defaults to alphanumeric sort
                            # Assuming standard structure: dataset/<plant>/<classes>
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
                        
                        # Apply confidence threshold logic for display
                        if confidence < confidence_threshold:
                            predicted_class_display = "Uncertain"
                        else:
                            predicted_class_display = predicted_class
                    else:
                        predicted_class = "Unknown"
                        predicted_class_display = "Unknown"

                    # Display
                    st.markdown(f"""
                    <div class="result-card">
                        <h2>Prediction: {predicted_class_display}</h2>
                        <h3 class="{ 'confidence-high' if confidence > confidence_threshold else 'confidence-low'}">
                            Confidence: {confidence * 100:.2f}%
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)

                    # Handing Low Confidence
                    if confidence < confidence_threshold:
                        st.markdown(f"""
                        <div class="warning-box">
                            ⚠️ <strong>Low Confidence Warning</strong><br>
                            The model is uncertain ({confidence*100:.1f}%). The image might be unclear, 
                            contain multiple leaves, or belong to an unsupported plant/disease type.
                            <br>Please consult an agricultural expert for confirmation.
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Treatment Recommendation
                    treatment = TREATMENTS.get(predicted_class, "Consult an agricultural expert for specific advice on this condition.")
                    
                    st.markdown(f"""
                    <div class="treatment-section">
                        <h4>💊 Recommended Treatment</h4>
                        <p>{treatment}</p>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.error("Model not found! Please train the model first by running `train.py`.")
    else:
        st.info("Please upload an image or use the camera to start analysis.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>© 2026 CropGuard AI • Hackathon Prototype</div>", unsafe_allow_html=True)
