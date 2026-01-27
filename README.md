# CropGuard AI 🌿

An AI-powered crop leaf disease detection system built for hackathons.

## 📄 Problem Statement
Agriculture faces significant losses due to plant diseases, often diagnosed too late by manual inspection. **CropGuard AI** provides a rapid, accessible solution for farmers to identify leaf diseases using just a smartphone camera, enabling timely treatment and protecting crop yields.

## 📂 Project Structure
- `dataset/`: Contains plant leaf images organized by class (disease).
- `models/`: Stores the trained model (`model.h5`) and class indices.
- `train.py`: Script to train the MobileNetV2 model.
- `app.py`: Streamlit web application for the demo.
- `requirements.txt`: Project dependencies.

## 📊 Dataset
The dataset is sourced from the **[New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)** on Kaggle.

> [!NOTE]
> For this prototype, we focus specifically on **Tomato crops**.

Due to repository size constraints and the large volume of high-resolution images, the raw dataset is **not included** in this git repository. Users should download the dataset from Kaggle and place it in the `dataset/` folder following the structure: `dataset/tomato/<class_name>/`.

### Classes Used
The model currently classifies the following 6 categories:
1. `Tomato_bac-spot` (Bacterial Spot)
2. `Tomato_Early-blight`
3. `Tomato_late-blight`
4. `tomato_leaf-mold`
5. `tomato_mosaic_virus`
6. `Tomato_Healthy`

## 🧠 Technical Approach
We utilize **Transfer Learning** with the **MobileNetV2** architecture.

- **Transfer Learning**: Instead of training from scratch, we leverage weights pretrained on ImageNet. This allows the model to "understand" shapes and textures immediately, requiring significantly less data and time to achieve high accuracy.
- **MobileNetV2**: Chosen for its high efficiency and low latency, making it ideal for mobile deployment or resource-constrained edge devices useful in field agriculture.

## 🎯 Why This Scope?
- **Time Constraints**: Focusing on a single crop (Tomato) ensures high reliability within a 24-48 hour hackathon window.
- **Reliability**: Deep specialization leads to more accurate specific diagnosis than a broad, shallow multi-crop model.
- **Prototype Focus**: This serves as a functional proof-of-concept that can be easily extended to other crops (Potato, Corn, etc.) using the same pipeline.

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
- **Frontend**: Streamlit (Web)
- **Processing**: OpenCV, Pillow, Numpy, Scipy

## 📝 Notes
- The model uses Transfer Learning (frozen base layers).
- Images are standardized to **224x224**.
- Confidence threshold is set to **65%** by default to prevent false positives.

