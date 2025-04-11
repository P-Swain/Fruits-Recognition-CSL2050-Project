import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import zipfile
from PIL import Image
from io import BytesIO

# Load model and preprocessor
model = tf.keras.models.load_model('fruit_ann_model.h5')

# Load the PCA transformer with 50 components
with open('pca_transformer.pkl', 'rb') as f:
    pca = pickle.load(f)

# Load the class labels
with open('class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)

# App UI config
st.set_page_config(page_title="Fruit Classifier", layout="wide")
st.title("🍎🍌 Fruit Recognition App")
st.write("Upload an image or a ZIP of images to classify fruits using ANN + PCA")

# Sidebar for input type
option = st.sidebar.selectbox("Select Input Type", ["Single Image", "Zip of Images"])

# Image preprocessing
def preprocess_image(img):
    img = img.resize((100, 100))  # Resize to match training input
    img_array = np.array(img) / 255.0  # Normalize
    img_flat = img_array.flatten().reshape(1, -1)  # Flatten
    img_pca = pca.transform(img_flat)  # Apply PCA
    return img_pca

# Prediction function
def predict(img_pca):
    probs = model.predict(img_pca)[0]
    label_idx = np.argmax(probs)
    return class_names[label_idx], probs[label_idx], probs

# Display predictions
def display_result(img, label, prob):
    st.image(img, caption=f"Prediction: {label} ({prob:.2f})", use_container_width=True)

# Main logic for single image or zip
if option == "Single Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        img_pca = preprocess_image(img)
        label, prob, _ = predict(img_pca)
        display_result(img, label, prob)

elif option == "Zip of Images":
    zip_file = st.file_uploader("Upload ZIP", type="zip")
    if zip_file:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            st.write("Processing Images...")
            for file_name in zip_ref.namelist():
                if file_name.endswith(('.jpg', '.jpeg', '.png')):
                    file_data = zip_ref.read(file_name)
                    img = Image.open(BytesIO(file_data)).convert("RGB")
                    img_pca = preprocess_image(img)
                    label, prob, _ = predict(img_pca)
                    st.image(img, caption=f"{file_name} → {label} ({prob:.2f})", use_column_width=True)

# Evaluation Results Section
st.markdown("---")
st.header("📊 Evaluation Results")
col1, col2 = st.columns(2)

with col1:
    st.image("grouped_confusion_matrix_named.png", caption="Grouped Confusion Matrix (Named)")
with col2:
    st.image("grouped_confusion_matrix_merged.png", caption="Grouped Confusion Matrix (Merged)")
