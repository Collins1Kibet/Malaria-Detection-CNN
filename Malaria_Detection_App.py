import json
from PIL import Image
import os

import numpy as np
import tensorflow as tf
import streamlit as st

pwd = os.getcwd()   # or  os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(pwd, 'Malaria_Detection_Model.h5')

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