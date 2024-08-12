# General Imports
import time
from datetime import datetime
import os
import numpy as np

# Tensorflow improrts
import tensorflow as tf
from keras.models import load_model
import keras.utils as utils
import PIL.Image as Image


# Retrieve environment variables else default value
data_path = os.getenv('DATA_PATH', '/mnt/data')
model_path = os.getenv('MODEL_PATH', '/mnt/save-model')
volume_mount_path = os.getenv('VOLUME_MOUNT_PATH', '/mnt')


# When actual image and model come in here need change
directpath = '/app/Toinput'
image_path = os.path.join(directpath, 'image.png')  # Assuming your image file is named 'image.png'
models_path = os.path.join(directpath, 'model_name.h5')  # Assuming your image file is named 'image.png'



# Function to return a log file object, ensuring path exists
def get_log_file():
    log_file_path = os.path.join(volume_mount_path, "inference.log")  # Use the environment variable    
    log = open(log_file_path, "a")    
    return log

# Starting logfile
log = get_log_file()
log.write("--------------------------\n")
log.write("Logging for inference pod\n")
log.write(f"Log start time: {datetime.now()}\n")
log.write("--------------------------\n")



# Loading of model
def load_model_from_path(models_path):
    """
    Load a Keras model from the specified path.

    Parameters:
    models_path (str): The file path to the saved Keras model.

    Returns:
    model (tensorflow.keras.Model): The loaded Keras model.
    
    Raises:
    Exception: If the model fails to load, an exception with the error details is raised.
    """
    try:
        model = load_model(models_path)
        print("Model loaded successfully.")
        log.write(f"{datetime.now()}: Model Loaded Successfully\n")
        return model
    except Exception as e:
        print(f"Failed to load the model. Error: {e}")
        log.write(f"{datetime.now()}: Failed to load the model. Error: {e}\n")


# Enusring model is loaded
try:
    model = load_model_from_path(models_path)
except:
    print(f"Failed to load the model")

# Ensuring image is loaded
try:
    img = utils.load_img(image_path, target_size=(64, 64))
    img_tensor = utils.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = img_tensor / 255.0
    print("Image loaded and preprocessed successfully.")
    log.write(f"{datetime.now()}: Image Loaded and preprocessed sucessfully\n")
except Exception as e:
    print(f"Failed to load or preprocess the image. Error: {e}")
    log.write(f"{datetime.now()}: Failed to load or preprocess the image. Error: {e}\n")



def get_predicted_class(prediction):
    """
    Get the predicted class label from the model's output.

    Parameters:
    prediction (np.ndarray): The model's output predictions, typically a softmax array.

    Returns:
    str: The predicted class label.
    """
    try:
        class_labels = {0: 'airplane', 1: 'automobile', 2: 'ship', 3: 'truck'}
        # Get the index of the highest predicted probability
        output = np.argmax(prediction)

        # Map the index to the class label
        predicted_class = class_labels.get(output, "Unknown class") 
        print(f"Successfully got predicted class")
        log.write(f"{datetime.now()}: Successfully got predicted class\n")
        return predicted_class

    except Exception as e:
        print(f"Failed to get predicted class. Error: {e}")
        log.write(f"{datetime.now()}: Failed to get predicted class. Error: {e}\n")


# Ensure Prediction
try:
    prediction = model.predict(img_tensor)
    output = get_predicted_class(prediction)
    print(f"Predicted class: {output}")
    log.write(f"{datetime.now()}: Model Predicted Successfully\n")
except Exception as e:
    print(f"Failed to predict image. Error: {e}")
    log.write(f"{datetime.now()}: Failed to predict image. Error: {e}\n")

# Closing logs
log.write("--------------------------\n")
log.write("Logging for inference pod\n")
log.write(f"Log end time: {datetime.now()}\n")
log.write("--------------------------\n\n")
log.close()

while True:
    time.sleep(5)  # Sleep for 5 seconds