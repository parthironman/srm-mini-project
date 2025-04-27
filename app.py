import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os

# Load YOLO classification model
@st.cache_resource
def load_yolo_model():
    return YOLO('best.pt')  # Replace with your trained classification model path

model = load_yolo_model()

# Function to classify image
def classify_image(image):
    # Save image to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name  # Store the path to use later

    try:
        # Run inference
        results = model(temp_path)

        # Get class names and probabilities
        names = results[0].names
        probs = results[0].probs.data.tolist()
        top_class_index = np.argmax(probs)
        top_class_name = names[top_class_index]
        top_score = probs[top_class_index]

        return top_class_name, top_score, probs, names

    finally:
        # Safely delete the temp file after inference is complete
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Streamlit UI
st.title("🧠 YOLO Image Classifier")
st.write("Upload an image and let the model classify it.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Classifying..."):
        label, confidence, all_probs, class_names = classify_image(image)

    st.success(f"**Prediction:** {label} ({confidence * 100:.2f}% confidence)")

    # Optional: Show full class probabilities
    if st.checkbox("Show all class probabilities"):
        st.write({class_names[i]: round(prob * 100, 2) for i, prob in enumerate(all_probs)})
