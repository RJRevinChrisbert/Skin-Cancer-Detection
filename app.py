import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

#   PAGE CONFIGURATION
st.set_page_config(
    page_title="Skin Cancer Detection – Multi-Class",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

#   CUSTOM CSS FOR UI
st.markdown("""
    <style>
        .main-title {
            font-size: 40px;
            font-weight: 700;
            text-align: center;
            color: #2E86C1;
        }
        .sub-title {
            font-size: 20px;
            text-align: center;
            color: #5D6D7E;
        }
        .result-box {
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
        }
        .green-light {
            background-color: #D4EFDF;
            color: #1D8348;
            border: 2px solid #27AE60;
        }
        .red-light {
            background-color: #F5B7B1;
            color: #922B21;
            border: 2px solid #C0392B;
        }
    </style>
""", unsafe_allow_html=True)

#   PAGE TITLE
st.markdown("<h1 class='main-title'>Skin Cancer Classification</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>· DenseNet201 Model · Multi-Class Prediction</p>", unsafe_allow_html=True)

#   LOAD MODEL
MODEL_PATH = "skinDiseaseDetectionUsningCNN.h5"

with st.spinner("Loading AI Model..."):
    model = load_model(MODEL_PATH)

#   CLASS LABELS (10 classes)
label_map = {
    0: "actinic keratosis",
    1: "basal cell carcinoma",
    2: "dermatofibroma",
    3: "melanoma",
    4: "nevus",
    5: "No Cancer",
    6: "pigmented benign keratosis",
    7: "seborrheic keratosis",
    8: "squamous cell carcinoma",
    9: "vascular lesion"
}

#   TRAIN MEAN & STD (REPLACE)
x_train_mean = 163.71742  
x_train_std = 41.23842    

#   SIDEBAR DETAILS
st.sidebar.header("Class Categories")
for k, v in label_map.items():
    st.sidebar.write(f"**{k} → {v}**")

#   FILE UPLOADER
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    
    # Show image preview
    st.subheader("Uploaded Image")
    img = Image.open(uploaded_file)
    st.image(img, caption="Preview", use_container_width=True)

    # Predict Button
    if st.button("🔍 Predict"):

        with st.spinner("Analyzing image... Please wait..."):
            time.sleep(2)

            # Prepare image
            model_h = model.input_shape[1]
            model_w = model.input_shape[2]

            img_resized = img.resize((model_w, model_h))
            img_array = np.asarray(img_resized).astype('float32')

            normalized = (img_array - x_train_mean) / x_train_std
            image_input = np.expand_dims(normalized, axis=0)

            # Prediction
            probabilities = model.predict(image_input)
            class_id = int(np.argmax(probabilities))
            confidence = float(np.max(probabilities)) * 100
            predicted_class = label_map[class_id]

        #   DISPLAY RESULT
        st.subheader("🔎 Prediction Result")

        # No Cancer → Green Light
        if predicted_class == "No Cancer":
            st.markdown(
                f"<div class='result-box green-light'>🟢 {predicted_class.upper()}</div>",
                unsafe_allow_html=True
            )
        else:
            # Any other class → Red light (cancer-related)
            st.markdown(
                f"<div class='result-box red-light'>🔴 {predicted_class.upper()}</div>",
                unsafe_allow_html=True
            )

        # Confidence bar
        st.subheader("📊 Confidence Level")
        st.progress(confidence / 100)
        st.write(f"**Confidence:** {confidence:.2f}%")

else:
    st.info("📤 Upload an image to begin prediction.")
