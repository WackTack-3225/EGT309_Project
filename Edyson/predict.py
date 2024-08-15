# General Imports
import time
from datetime import datetime
import os
import numpy as np
from flask import Flask, request, jsonify
import io
import base64


# Tensorflow Imports
import tensorflow as tf
from keras.models import load_model
import keras.utils as utils
import PIL.Image as Image

app = Flask(__name__)


# Retrieve environment variables or else use default path value
model_path = os.getenv('MODEL_PATH', '/mnt/saved_model')


# To join the model_path which is where the volume with the saved model is mounted at with the model file
saved_model_path = os.path.join(model_path, 'trainedmodel.h5') 


#### General functions
# General function 1: For logging output through creation of a logging file (logging file is saved in the volume for persistency)
def get_log_file():
    log_file_path = os.path.join(model_path, "inference.log")  # Use the environment variable    
    log = open(log_file_path, "a")    
    return log



# General function 2: To get label of class through mapping
def get_predicted_class(prediction):
    try:
        class_labels = {0: 'airplane', 1: 'automobile', 2: 'ship', 3: 'truck'}
        output = np.argmax(prediction)
        predicted_class = class_labels.get(output, "Unknown class")
        print(f"Successfully got predicted class")
        return predicted_class
    except Exception as e:
        print(f"Failed to get predicted class. Error: {e}")



# General function 3: Loading of model base on model path
def load_model_from_path(models_path):
    try:
        model = load_model(models_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Failed to load the model. Error: {e}")

### App route function that utilizes the general functions
# Flask route to handle image prediction when a POST request via RESTful API containing a JSON Payload with Base64 encoding images is send with return of results in the same base64 format for display
@app.route('/predict', methods=['POST'])
def predict():
    # Initalizing log file
    log = get_log_file()
    log.write("--------------------------\n")
    log.write("Logging for inference pod\n")
    log.write(f"Log start time: {datetime.now()}\n")
    log.write("--------------------------\n")
    
    # Extract data in request which is the JSON payload sent from the web-app
    data = request.get_json()

    # Check if the JSON Payload matches the data it is intended to handle which is a JSON Payload of base64 enconded images with exception handling
    if 'images' not in data:
        log.write(f"{datetime.now()}: No image files in request\n")
        return jsonify({"error": "No image files in request"}), 400

    # Extraction of image data
    images_data = data['images']
    response_list = []

    # Load the model from define path with exception handling
    model = load_model_from_path(saved_model_path)
    if model is None:
        return jsonify({"error": "Model could not be loaded"}), 500

    # Looping through each incoming base64-encoded images from a JSON payload, converts them to image tensors
    #  (the required format for the model to predict), run them through a machine learning model for prediction, and prepares the results as JSON responses for display on a web interface. 
    for image_data_dict in images_data:
        if 'image' not in image_data_dict:
            continue  # Skip if 'image' key is not present (meaning it is not the format expected)
        
        # Extraction of base64-encoded image data from the dictionary
        image_data = image_data_dict['image']

        # Processing image ensuring it is ready for model prediction with logging
        try:
            # Decode the base64 image data into raw bytes
            image_bytes = base64.b64decode(image_data)

            # Open the image from the decoded bytes using Pillow and resize
            img = Image.open(io.BytesIO(image_bytes))
            img = img.resize((64, 64))

            # Convert the image to a NumPy array (tensor) for processing
            img_tensor = np.array(img)

            # Expand dimensions of tensor add a batch dimension (from [64, 64, 3] to [1, 64, 64, 3]) with normalization
            img_tensor = np.expand_dims(img_tensor, axis=0)
            img_tensor = img_tensor / 255.0

            log.write(f"{datetime.now()}: Image received and preprocessed successfully\n")
        except Exception as e:
            log.write(f"{datetime.now()}: Failed to load or preprocess the image. Error: {e}\n")
            continue # skip to next image

        # Ensure model is able to predict the image with saving of results to return to webpage for display
        try:
            # Sending image to model for prediction with extraction of results and confidence score
            prediction = model.predict(img_tensor)
            predicted_class  = get_predicted_class(prediction)
            confidence = np.max(prediction)  # Get the confidence score (probability 0 to 1)
            print(f"Predicted class: {predicted_class }")

            # Create response object for this image, saving its base64-encoded image data (must be base64 to exchange image data), prediction, and confidence score to return to the webpage
            response = {
                "imageUrl": f"data:image/png;base64,{image_data}",
                "prediction": predicted_class,
                "confidence": round(float(confidence), 2)
            }
            response_list.append(response)

            # Logging of predicted result for validation and traceability
            log.write(f"{datetime.now()}: Model Predicted Successfully of class {predicted_class}\n")
        except Exception as e:
            print(f"Failed to predict image. Error: {e}")
            log.write(f"{datetime.now()}: Failed to predict image. Error: {e}\n")
            continue

    # Closing logs saving the logfile for validation, traceability and review of results of the POST request
    log.write("--------------------------\n")
    log.write("Logging for inference pod\n")
    log.write(f"Log end time: {datetime.now()}\n")
    log.write("--------------------------\n\n")
    log.close()

    return jsonify(response_list)

# Running the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8083)