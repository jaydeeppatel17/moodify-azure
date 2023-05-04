from flask import Flask, request, jsonify
import cv2
import numpy as np
from keras.models import load_model
import json
import streamlit as st

# Load the model
moodDetector = load_model("moodifyEngine.h5")

# Define a dictionary to map class indices to emotion labels
label_dictionary = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}

# Define a helper function to preprocess the image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    return normalized.reshape((1,48,48,1))

# Define the prediction function
def predict(image):
    # Preprocess the image
    input_data = preprocess_image(image)
    
    # Make a prediction using the model
    prediction = np.argmax(moodDetector.predict(input_data), axis=-1)
    
    # Get the corresponding emotion label
    emotion = label_dictionary.get(prediction[0])
    
    return emotion

# Create a Flask app
app = Flask(__name__)

# Define an endpoint for handling image uploads
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    
    # Read the image as a numpy array
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Make a prediction
    emotion = predict(image)
    
    # Return the emotion as a JSON response
    response = {'emotion': emotion}
    return response

# Run the app
if __name__ == '__main__':
    st.set_page_config(page_title='Moodify Engine')
    st.title('Moodify Engine')
    st.write('Upload an image to predict the emotion')
    app.run()
