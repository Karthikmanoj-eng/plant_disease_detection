import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Set up page configurations
st.set_page_config(page_title="group 7 Plant Disease Classifier", layout="centered")

# --- 1. LOAD THE TRAINED MODEL ---
@st.cache_resource
def load_plant_model():
    # Updated to point to your new Tomato-Optimized EfficientNetB0 file name!
    return tf.keras.models.load_model('model65.keras')

try:
    model = load_plant_model()
except Exception as e:
    st.error(f"Error loading model file. Make sure 'best.keras' is in your directory. Details: {e}")
    st.stop()

# --- 2. DEFINE THE EXACT 28 PLANTDOC CLASSES (FIXED ORDER) ---
CLASS_NAMES = [
    'Apple_Scab_Leaf', 'Apple_leaf', 'Apple_rust_leaf', 'Bell_pepper_leaf', 
    'Bell_pepper_leaf_spot', 'Blueberry_leaf', 'Cherry_leaf', 'Corn_Gray_leaf_spot', 
    'Corn_leaf_blight', 'Corn_rust_leaf', 'Peach_leaf', 'Potato_leaf_early_blight', 
    'Potato_leaf_late_blight', 'Raspberry_leaf', 'Soyabean_leaf', 'Squash_Powdery_mildew_leaf', 
    'Strawberry_leaf', 'Tomato_Early_blight_leaf', 'Tomato_Septoria_leaf_spot', 'Tomato_leaf', 
    'Tomato_leaf_bacterial_spot', 'Tomato_leaf_late_blight', 'Tomato_leaf_mosaic_virus', 
    'Tomato_leaf_yellow_virus', 'Tomato_mold_leaf', 'Tomato_two_spotted_spider_mites_leaf', 
    'grape_leaf', 'grape_leaf_black_rot'
]

# Clean up names for a prettier UI display
def clean_label(label):
    return label.replace('_', ' ')

# --- 3. USER INTERFACE LAYOUT ---
st.title("🌿 PlantDoc Disease Detector")
st.write("Upload a leaf image or take a photo to instantly identify plant species and potential diseases.")

# --- SIDEBAR: WHAT IT CAN DETECT ---
with st.sidebar:
    st.header("📋 Supported Target Classes")
    st.write(f"This system is trained to identify **{len(CLASS_NAMES)} distinct classes**:")
    for name in sorted(CLASS_NAMES):
        st.markdown(f"- {clean_label(name)}")

# --- 4. PHOTOGRAPHY GUIDELINES ---
st.subheader("📸 Photo Best Practices")
st.info("""
For the highest accuracy, please follow these steps when capturing your photo:
1. **Isolate a Single Leaf:** Keep a single leaf centered in the frame. Avoid clusters or messy branches.
2. **Focus on the Infection:** Ensure any visible spots, rust, or discoloration are sharp and in focus.
3. **Minimize Background Noise:** Avoid capturing excessive soil, weeds, or human hands holding the leaf.
4. **Natural Lighting:** Ensure the leaf is evenly lit. Heavy shadows can scramble color profiles.
""")

# --- 5. IMAGE INPUT (UPDATED TO ALLOW WEBP IMAGES) ---
img_file_buffer = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png", "webp"])

if img_file_buffer is not None:
    # Display the uploaded image
    image = Image.open(img_file_buffer)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("⚙️ *Analyzing image textures and features...*")
    
    # --- 6. PREPROCESSING TO MATCH EFFICIENTNETB0 ---
    # Resize to target dimensions (224x224)
    image_resized = image.resize((224, 224))
    
    # Convert PIL image to numpy array
    img_array = tf.keras.utils.img_to_array(image_resized)
    
    # Add batch dimension (Shape becomes [1, 224, 224, 3])
    img_array = np.expand_dims(img_array, axis=0)

    # --- 7. INFERENCE ---
    predictions = model.predict(img_array)
    
    # Cleanly convert raw logits output from from_logits=True pipeline to probability distributions
    probabilities = tf.nn.softmax(predictions[0]).numpy()
    best_class_idx = np.argmax(probabilities)
    confidence = probabilities[best_class_idx] * 100
    
    raw_label = CLASS_NAMES[best_class_idx]
    predicted_label = clean_label(raw_label)
    
    # --- 8. DISPLAY RESULTS ---
    st.subheader("🔬 Diagnostic Results")
    
    # Check if the name implies a baseline healthy leaf or a specific condition
    is_healthy = True
    if "spot" in raw_label.lower() or "blight" in raw_label.lower() or "rust" in raw_label.lower() or "scab" in raw_label.lower() or "mildew" in raw_label.lower() or "virus" in raw_label.lower() or "mold" in raw_label.lower() or "mites" in raw_label.lower() or "rot" in raw_label.lower():
        is_healthy = False

    if confidence > 35:  # Realistic threshold for noisy datasets
        if is_healthy:
            st.success(f"**Result:** Genus identified as **{predicted_label}** ({confidence:.2f}% confidence). Visual markers indicate a healthy leaf structure.")
        else:
            st.error(f"**Detected Condition:** {predicted_label} ({confidence:.2f}% confidence)")
            st.warning("⚠️ **Recommendation:** Isolate the affected plant if possible, ensure proper air circulation to reduce humidity, and check targeted treatments for this specific disease strain.")
    else:
        st.warning(f"Low confidence warning ({confidence:.2f}%). The model tentatively suspects **{predicted_label}**, but the picture may be too blurry, cropped poorly, or noisy. Try taking another shot following the best practices above.")
