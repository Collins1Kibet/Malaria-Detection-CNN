import json
from PIL import Image
import os

import gdown
import numpy as np
import tensorflow as tf
import streamlit as st

def download_model(drive_url, output_path):
    if not os.path.exists(output_path):
        gdown.download(drive_url, output_path, quiet=False)

drive_url = 'https://drive.google.com/uc?id=1bwAzohfaBkwJaifU-hs57thCea5cPyx8'
model_path = 'Malaria_Detection_Model.h5'

# Downloading the model
download_model(drive_url, model_path)

# Loading the model
model = tf.keras.models.load_model(model_path)

def load_and_preprocess_image(image, target_size=(135, 135)):
    try:
        image = Image.open(image)
        image = image.resize(target_size)
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array.astype('float32') / 255.
        return image_array
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def predict_image(model, image):
    preprocessed_image = load_and_preprocess_image(image)
    if preprocessed_image is not None:
        prediction = model.predict(preprocessed_image)
        return 'Uninfected' if prediction[0] > 0.5 else 'Infected'
    else:
        return 'Error'

st.title('🦟 Malaria Detection System')

uploaded_image = st.file_uploader('Upload Cell Image...', type=['jpeg', 'jpg', 'png'])

if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image)

        col1, col2 = st.columns(2)

        with col1:
            resized_image = image.resize((150, 150))
            st.image(resized_image)

        with col2:
            if st.button('Detect Malaria'):
                prediction = predict_image(model, uploaded_image)
                st.success(f'Prediction: {str(prediction)}')
    except Exception as e:
        st.error(f"Error loading image: {e}")
