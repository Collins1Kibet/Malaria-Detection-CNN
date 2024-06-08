import json
from PIL import Image
import os

import gdown
import numpy as np
import tensorflow as tf
import streamlit as st

def download_model(drive_url, output_path):
    if not os.path.exists(output_path):
        print(f"Downloading model from {drive_url}")
        gdown.download(drive_url, output_path, quiet=False)
    else:
        print(f"Model file already exists at {output_path}")

# Google Drive link to model file
drive_url = 'https://drive.google.com/uc?id=1bwAzohfaBkwJaifU-hs57thCea5cPyx8'
model_path = 'Malaria_Detection_Model.h5'

# Download the model
download_model(drive_url, model_path)

# Check if the file exists and its size
if os.path.exists(model_path):
    print(f"Model file downloaded successfully, size: {os.path.getsize(model_path)} bytes")
else:
    print("Model file was not downloaded successfully.")

download_model(drive_url, model_path)
model = tf.keras.models.load_model(model_path)

def load_and_preprocess_image(image_path, target_size=(135, 135)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array.astype('float32') / 255.
    return image_array

def predict_image(model, image_path):
    preprocessed_image = load_and_preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    return 'Uninfected' if prediction[0] > 0.5 else 'Infected'

# Modelling the Streamlit Web app Page
st.title('🦟 Malaria Detection System')

uploaded_image = st.file_uploader('Upload Cell Image...', type=['jpeg', 'jpg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    col1, col2 = st.columns(2)

    with col1:
        resized_image = image.resize((150, 150))
        st.image(resized_image)

    with col2:
        if st.button('Detect Malaria'):
            prediction = predict_image(model, uploaded_image)
            st.success(f'Prediction: {str(prediction)}')
