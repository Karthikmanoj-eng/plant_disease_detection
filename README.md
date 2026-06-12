# Plant Disease Detection

An end-to-end Deep Learning project that utilizes Transfer Learning with **MobileNetV2** via TensorFlow/Keras to classify plant diseases across 28 distinct categories using the **PlantDoc Dataset**. 

This model achieves a classification accuracy of **65%** across highly varied, real-world field conditions.

## 🚀 Features
- **Data Augmentation**: Built-in robust pre-processing layer pipeline including Random Flips, Rotations, and Zoom modifications to prevent overfitting.
- **Transfer Learning**: Built on top of pre-trained ImageNet weights from **MobileNetV2**, optimizing computational efficiency while preserving highly generalized visual accuracy.
- **Automated Model Checkpointing**: Tracks evaluation loss during training epochs and captures the highest performance configurations automatically.
- **Interactive Web App**: A lightweight, user-friendly Streamlit application interface to upload plant leaf images and view real-time classification predictions instantly.

## 📊 Dataset Detail
The pipeline utilizes the **PlantDoc Dataset** fetched via Kaggle API containing **28 architectural botanical classes**. Examples include:
- `Apple_Scab_Leaf`, `Apple_rust_leaf`
- `Corn_Gray_leaf_spot`, `Corn_rust_leaf`
- `Potato_leaf_early_blight`, `Potato_leaf_late_blight`
- `Tomato_leaf_bacterial_spot`, `Tomato_leaf_yellow_virus`, and more.

## 🛠️ Tech Stack & Architecture
- **Framework**: TensorFlow 2.x / Keras
- **Deployment**: Streamlit
- **Base Architecture**: MobileNetV2 (Frozen Base)
- **Image Target Dimensions**: 224 x 224 pixels
- **Batch Size**: 32
- **Classification Output Dense Layer**: 28 Units with Softmax Activation

---

## 🔧 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/Karthikmanoj-eng/plant_disease_detection.git](https://github.com/Karthikmanoj-eng/plant_disease_detection.git)
cd plant_disease_detection
