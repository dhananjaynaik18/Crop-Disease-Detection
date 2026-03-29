import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import tensorflow as tf
import datetime
import os
import time

# ADVANced UI Set page configuration
st.set_page_config(page_title="Crop AI", page_icon="🌿", layout="wide")

# LOAD model
model = tf.keras.models.load_model("model/trained_model.h5")

# Class labels (Must match your data folders exactly)
classes = [
    "Potato_Early_blight",
    "Potato_healthy",
    "Tomato_Late_blight",
    "Tomato_healthy"
]

# Disease info
disease_info = {
    "Tomato_Late_blight": {
        "description": "Fungal disease causing dark, irregular spots on tomato leaves.",
        "treatment": "Use fungicides like Mancozeb. Avoid overhead watering to keep leaves dry."
    },
    "Potato_Early_blight": {
        "description": "Brown spots with concentric rings appearing on older leaves.",
        "treatment": "Apply chlorothalonil fungicide and ensure proper crop rotation."
    }
}

history_file = "prediction_history.csv"

def preprocess(image):
    img = np.array(image)
    img = cv2.resize(img, (128, 128))
    img=img/255.0                    ###Delete this line because the image will twice divided by 255 here also and in app also##
    img = np.expand_dims(img, axis=0)
    return img

def save_history(data):
    df = pd.DataFrame([data])
    if os.path.exists(history_file):
        df.to_csv(history_file, mode='a', header=False, index=False)
    else:
        df.to_csv(history_file, index=False)

# ADVANCED UI: Sidebar setup
st.sidebar.title("🌿 AI Dashboard")
st.sidebar.info("Upload a picture of a crop leaf to detect diseases instantly using Deep Learning.")
if os.path.exists("model/accuracy.png"):
    st.sidebar.image("model/accuracy.png", caption="AI Training Accuracy")

# Main Title
st.title("🌾 Crop Disease Detection System")
st.markdown("Identify plant health issues quickly with our custom-trained CNN.")

uploaded_file = st.file_uploader("Upload Leaf Image (JPG/PNG)", type=["jpg", "png"])

if uploaded_file:
    # ADVANCED UI: Side-by-side columns
    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        if st.button("🔍 Run AI Analysis", use_container_width=True):
            
            # ADVANCED UI: Loading spinner for realistic feel
            with st.spinner("Analyzing leaf patterns..."):
                time.sleep(1) # Adds a slight processing delay for user experience
                processed = preprocess(image)
                prediction = model.predict(processed)
                class_index = np.argmax(prediction)
                confidence = np.max(prediction)
                label = classes[class_index]

            # Display formatted label
            display_name = label.replace("_", " ")
            st.success(f"**Diagnosis:** {display_name}")

            # ADVANCED UI: Visual progress bar for confidence
            st.write("**AI Confidence Level:**")
            st.progress(float(confidence))
            st.write(f"{confidence*100:.2f}%")

            # Display Treatment
            if "healthy" in label.lower():
                st.balloons() # Fun animation for a healthy plant
                st.success("This plant appears to be completely healthy! 🌱")
            elif label in disease_info:
                st.info(f"💡 **Treatment:** {disease_info[label]['treatment']}")
                st.write(f"**Description:** {disease_info[label]['description']}")

            # Save to history
            save_history({
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "prediction": display_name,
                "confidence": round(float(confidence), 4)
            })

# ADVANCED UI: Expandable history block
st.write("---")
with st.expander("📜 View Prediction History"):
    if os.path.exists(history_file):
        history = pd.read_csv(history_file)
        st.dataframe(history, use_container_width=True)
    else:
        st.write("No predictions made yet.")
