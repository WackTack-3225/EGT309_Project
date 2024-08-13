# General Imports
import time
from datetime import datetime
import os
import numpy as np
from flask import Flask, request, jsonify
import io
import base64


# Tensorflow improrts
import tensorflow as tf
from keras.models import load_model
import keras.utils as utils
import PIL.Image as Image

app = Flask(__name__)


# Retrieve environment variables else default value
data_path = os.getenv('DATA_PATH', '/mnt/data')
model_path = os.getenv('MODEL_PATH', '/mnt/save-model')
volume_mount_path = os.getenv('VOLUME_MOUNT_PATH', '/mnt')


# When actual image and model come in here need change
directpath = '/app/Toinput'
image_path = os.path.join(directpath, 'image.png')  # Assuming your image file is named 'image.png'
models_path = os.path.join(directpath, 'model_name.h5')  # Assuming your image file is named 'image.png'


#### General functions
# General function 1: For logging output
def get_log_file():
    log_file_path = os.path.join(volume_mount_path, "inference.log")  # Use the environment variable    
    log = open(log_file_path, "a")    
    return log



# General function 2: To get label of class
def get_predicted_class(prediction):
    try:
        class_labels = {0: 'airplane', 1: 'automobile', 2: 'ship', 3: 'truck'}
        output = np.argmax(prediction)
        predicted_class = class_labels.get(output, "Unknown class")
        print(f"Successfully got predicted class")
        return predicted_class
    except Exception as e:
        print(f"Failed to get predicted class. Error: {e}")



# General function 3: Loading of model
def load_model_from_path(models_path):
    try:
        model = load_model(models_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Failed to load the model. Error: {e}")


# Flask route to handle image prediction
@app.route('/predict', methods=['POST'])
def predict():
    log = get_log_file()
    log.write("--------------------------\n")
    log.write("Logging for inference pod\n")
    log.write(f"Log start time: {datetime.now()}\n")
    log.write("--------------------------\n")
    

    data = request.get_json()

    # Check if an image file is included in the request
    if 'images' not in data:
        log.write(f"{datetime.now()}: No image files in request\n")
        return jsonify({"error": "No image files in request"}), 400

    images_data = data['images']
    response_list = []

    # Load the model
    model = load_model_from_path(models_path)
    if model is None:
        return jsonify({"error": "Model could not be loaded"}), 500

    for image_data_dict in images_data:
        if 'image' not in image_data_dict:
            continue  # Skip if 'image' key is not present
        
        image_data = image_data_dict['image']

        try:
            # Decode the base64 image data
            image_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(image_bytes))
            img = img.resize((64, 64))
            img_tensor = np.array(img)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            img_tensor = img_tensor / 255.0
            log.write(f"{datetime.now()}: Image received and preprocessed successfully\n")
        except Exception as e:
            log.write(f"{datetime.now()}: Failed to load or preprocess the image. Error: {e}\n")
            continue # skip to next image

        # Ensure Prediction
        try:
            prediction = model.predict(img_tensor)
            predicted_class  = get_predicted_class(prediction)
            confidence = np.max(prediction)  # Get the confidence (maximum probability)
            print(f"Predicted class: {predicted_class }")

            # Create response object for this image
            response = {
                "imageUrl": f"data:image/png;base64,{image_data}",
                "prediction": predicted_class,
                "confidence": round(float(confidence), 2)
            }
            response_list.append(response)

            log.write(f"{datetime.now()}: Model Predicted Successfully of class {predicted_class}\n")
        except Exception as e:
            print(f"Failed to predict image. Error: {e}")
            log.write(f"{datetime.now()}: Failed to predict image. Error: {e}\n")
            continue

    # Closing logs
    log.write("--------------------------\n")
    log.write("Logging for inference pod\n")
    log.write(f"Log end time: {datetime.now()}\n")
    log.write("--------------------------\n\n")
    log.close()

    return jsonify(response_list)

# Running the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8083)