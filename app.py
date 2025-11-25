import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# -------------------------------
# Load the trained model
# -------------------------------
model = tf.keras.models.load_model("skinDiseaseDetectionUsningCNN.h5")

# Your label map (IMPORTANT – use your project’s mapping)
label_map = {
    0: "Class_0",
    1: "Class_1",
    2: "Class_2",
    3: "Class_3",
    4: "Class_4",
    5: "Class_5",
    6: "Class_6",
    7: "Class_7",
    8: "Class_8",
    9: "Class_9"
}

# ---------------------------------------
# Streamlit Page Configuration
# ---------------------------------------
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------
# Custom Dark Theme UI
# ---------------------------------------
st.markdown("""
    <style>
        .main {
            background-color: #0f1225;
            color: white;
        }
        .stButton>button {
            background: linear-gradient(45deg,#4e3df5,#a22ef8);
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .upload-box {
            padding: 20px;
            border: 2px dashed #666;
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            text-align: center;
        }
        .prediction-box {
            padding: 20px;
            border-radius: 10px;
            background: rgba(255,255,255,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------
# Title
# ---------------------------------------
st.markdown("<h1 style='text-align:center;'>🩺 Skin Cancer Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; opacity:0.8;'>Upload an image to classify the disease.</p>", unsafe_allow_html=True)

# ---------------------------------------
# Layout: Left → Upload | Right → Image + Result
# ---------------------------------------
col1, col2 = st.columns([1,2])

with col1:
    st.markdown("<div class='upload-box'>Click below to upload skin image</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    predict_button = st.button("Predict")

with col2:
    st.subheader("Image Preview")
    image_container = st.empty()

    st.subheader("Prediction Result")
    result_box = st.empty()

# ---------------------------------------
# Prediction Logic
# ---------------------------------------
if predict_button:
    if uploaded_file is None:
        st.warning("⚠ Please upload an image first.")
    else:
        # Load image
        img = Image.open(uploaded_file)
        
        # Model expects (75,100) sized images (as per your project)
        img_resized = img.resize((100, 75))
        img_array = np.array(img_resized)

        # Normalize (IMPORTANT — match your model)
        img_norm = (img_array - np.mean(img_array)) / np.std(img_array)
        img_norm = np.expand_dims(img_norm, axis=0)

        # Predict
        predictions = model.predict(img_norm)
        class_index = np.argmax(predictions)
        confidence = np.max(predictions) * 100
        predicted_label = label_map[class_index]

        # Display image
        image_container.image(img, caption="Uploaded Image", use_column_width=True)

        # Display result
        result_box.markdown(f"""
        <div class='prediction-box'>
            <h3>🔍 Prediction: <span style='color:#a97dfc'>{predicted_label}</span></h3>
            <h4>Confidence: {confidence:.2f}%</h4>
        </div>
        """, unsafe_allow_html=True)
