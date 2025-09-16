import streamlit as st
import tensorflow as tf
from keras.preprocessing import image
from PIL import Image
import numpy as np
import cv2

# Load the trained model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("my_model.h5")

model = load_model()

# Function to map prediction code to class name
def Transfername(namecode):
    return "Benign" if namecode == 0 else "Malignant"

# Preprocess uploaded image
def preprocess_image(uploaded_file):
    im = Image.open(uploaded_file).convert("RGB")
    im = im.resize((224, 224))  # use PIL resize instead of cv2
    img = np.array(im) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# Predict using the model
def predict_output(img):
    pred_result = model.predict(img)
    result = 1 if pred_result[0][0] > 0.5 else 0
    return result

# Streamlit UI
st.title("Skin Cancer Classification App 🩺")
st.write("Upload an image to classify as **Benign** or **Malignant**")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    img = preprocess_image(uploaded_file)
    prediction = predict_output(img)
    label = Transfername(prediction)

    st.subheader(f"Prediction: {label}")
