from flask import Flask, request, render_template, jsonify
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io
import datetime
import base64

import webbrowser

# Library for model
import bahan

app = Flask(__name__)

# Custom objects required for loading the model
custom_objects = {
    'ConvBlock': bahan.ConvBlock,
    'UNet': bahan.UNet,
    'dice_coefficient': bahan.dice_coefficient,
    'ActivationLayer': bahan.ActivationLayer
}

# Load the model
model = load_model("weight_updated.h5", custom_objects=custom_objects)

# Image size expected by the model
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Directory to save images
SAVE_DIR = 'saved_images'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

@app.route('/')
def upload_file():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict(debug="False"):
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if debug: print("[DEBUG]: ",str(type(file)))
    if file.filename == '':
        return "No selected file"
    
    if file:
        # Generate a unique filename based on the current time
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = f"{SAVE_DIR}/original_{timestamp}.png"
        predicted_filename = f"{SAVE_DIR}/predicted_{timestamp}.png"

        # Convert the file to a BytesIO object
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

        # Save the original image
        img.save(original_filename)

        # Resize the image to match the model's input size
        img = img.resize((IMG_HEIGHT, IMG_WIDTH))

        # Convert the image to an array
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict the segmentation mask
        prediction = model.predict(img_array)
        prediction = np.squeeze(prediction, axis=0)

        # Choose the class with the highest probability
        segmentation_map = np.argmax(prediction, axis=-1)  # shape becomes (256, 256)
        if debug: print(segmentation_map)

        # Convert the prediction to an image
        predicted_img = Image.fromarray((segmentation_map * 255).astype('uint8'))

        # Save the predicted image
        predicted_img.save(predicted_filename)

        # Save the image to a BytesIO object to send back as a response
        img_io = io.BytesIO()
        predicted_img.save(img_io, 'PNG')
        img_io.seek(0)
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

        return jsonify({'image': img_base64})

if __name__ == "__main__":
    webbrowser.open('http://127.0.0.1:5000/')
    app.run(debug=True)
    