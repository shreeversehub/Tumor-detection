import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

IMG_SIZE = 128

# ----------------------------
# Load trained model
# ----------------------------
@st.cache_resource
def get_model():
    return load_model("model/cnn_model.h5")

model = get_model()

# ----------------------------
# Preprocess uploaded image
# ----------------------------
def preprocess_image(image):
    img = np.array(image.convert("L"))  # convert to grayscale
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return img

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠")

st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI scan to check for the presence of a tumor.")

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

    if st.button("Analyze"):
        with st.spinner("Analyzing scan..."):
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)[0][0]

        if prediction > 0.5:
            confidence = prediction * 100
            st.error(f"⚠️ Tumor Detected — Confidence: {confidence:.2f}%")
        else:
            confidence = (1 - prediction) * 100
            st.success(f"✅ No Tumor Detected — Confidence: {confidence:.2f}%")

        st.caption("This tool is for educational purposes only and is not a substitute for professional medical diagnosis.")
