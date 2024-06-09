## 🦟 Malaria Detection System

This is a web application built with Streamlit that allows users to detect malaria from cell images. The app utilizes a pre-trained TensorFlow model to classify cell images as either 'Infected' or 'Uninfected'.

### Features

- Upload a cell image in JPEG, JPG, or PNG format.
- Automatically downloads the pre-trained model from Google Drive if not already present.
- Displays the uploaded image.
- Provides a prediction on whether the cell is infected with malaria.

### Requirements

- Python 3.6+
- Streamlit
- TensorFlow
- gdown
- Pillow
- NumPy

### Building, Training, and Testing the Model
The Jupyter Notebook used to build, train, and test the model is included in this repository as Malaria_Detection.ipynb. The dataset used is from Kaggle: [Cell Images for Detecting Malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria).

### Code Overview
- download_model(drive_url, output_path): Function to download the pre-trained model if it's not already present in the specified path.
- load_and_preprocess_image(image, target_size=(135, 135)): Function to load and preprocess the uploaded image to the required format for the model.
- predict_image(model, image): Function to predict whether the uploaded cell image is 'Infected' or 'Uninfected' using the pre-trained model.
*** Streamlit app setup: ***
- Title and description of the app.
- File uploader for the user to upload cell images.
- Display the uploaded image.
- Button to trigger malaria detection.
- Display the prediction result.
