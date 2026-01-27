# CropGuard AI 🌿

An AI-powered crop leaf disease detection system built for hackathons.

## 🎯 Project Goal
Identify crop leaf diseases from images using Deep Learning (MobileNetV2) and provide treatment recommendations.

## 📂 Structure
- `dataset/`: Contains plant leaf images organized by class (disease).
- `models/`: Stores the trained model (`model.h5`) and class indices.
- `train.py`: Script to train the MobileNetV2 model.
- `app.py`: Streamlit web application for the demo.
- `requirements.txt`: Project dependencies.

## 🚀 Setup & Run

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model**
   Run the training script to generate `models/model.h5` and `models/class_indices.txt`.
   ```bash
   python train.py
   ```

3. **Run the App**
   Launch the Streamlit interface.
   ```bash
   streamlit run app.py
   ```

## 🛠 Tech Stack
- **AI/ML**: TensorFlow, Keras, MobileNetV2
- **Frontend**: Streamlit
- **Processing**: OpenCV, Pillow, Numpy

## 📝 Notes
- The model uses Transfer Learning (frozen base layers).
- Images are resized to 224x224.
- Confidence threshold is set to 65% by default.
