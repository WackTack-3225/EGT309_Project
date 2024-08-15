import os
import sys
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from threading import Thread


from flask import Flask, request, jsonify, render_template
import base64
import json
import subprocess
import requests

app = Flask(_name_)

@app.route('/train', methods=['POST'])
def train():
    # Retrieve the status code from the request headers
    status_code = request.headers.get('Status-Code', type=int)  # Typecast to int

    # Check if the status code is 200
    if status_code != 200:
        return jsonify({"error": "Invalid status code. Expected 200."}), 400

    # Start the training process in a new thread
    thread = Thread(target=run_training_and_notify)
    thread.start()

    # Immediately return a response indicating that training has started
    return jsonify({"status": "Training startings"}), 200
    



def run_training_and_notify():
    # main model training section
    try:
        # Define the paths to the training and validation data
        train_data_dir = '/app/data/train'
        validation_data_dir = '/app/data/test'
        with open('/app/error_log.txt', "a") as log:
            log.write("Threading start\n")

        # Ensure the directories exist
        if not os.path.exists(train_data_dir):
            raise FileNotFoundError(f"Training data directory not found: {train_data_dir}")
        if not os.path.exists(validation_data_dir):
            raise FileNotFoundError(f"Validation data directory not found: {validation_data_dir}")
        
        # Function to check the number of images in each class directory
        def check_images_in_directory(directory):
            for dirpath, dirnames, filenames in os.walk(directory):
                if dirnames:  # If it's a directory with subdirectories
                    print(f"Checking directory: {dirpath}")
                else:
                    print(f"Found {len(filenames)} images in {dirpath}")

        # Check the train directory
        print("Checking training data directory...")
        check_images_in_directory(train_data_dir)

        # Check the test directory
        print("Checking validation data directory...")
        check_images_in_directory(validation_data_dir)

        with open('/app/error_log.txt', "a") as log:
            log.write("Validate train and validation paths\n")


        # Load the training and validation data
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            train_data_dir,
            image_size=(64, 64),  # Adjust size as needed
            batch_size=100,
            label_mode='categorical'  # Assuming categorical labels
        )

        validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            validation_data_dir,
            image_size=(64, 64),  # Adjust size as needed
            batch_size=100,
            label_mode='categorical'  # Assuming categorical labels
        )




        # Path to save the model
        model_save_path = os.getenv('MODEL_SAVE_PATH', '/mnt/saved_model')

        if not model_save_path:
            raise ValueError("MODEL_SAVE_PATH environment variable is not set or is empty.")
        
        # Ensure the directory exists where the model will be saved
        model_save_dir = os.path.dirname(model_save_path)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)


        with open('/app/error_log.txt', "a") as log:
            log.write("Create and validate model saving path\n")

        # MobileNetV2 Model as base model
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(64, 64, 3))


        # Modify input shape and classifier layers
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(2, activation='relu')(x)
        predictions = layers.Dense(4, activation='softmax')(x)
        transfer_model = models.Model(inputs=base_model.input, outputs=predictions)


        # Freezing the first 10 layers of the model
        base_model.trainable = True
        for layer in transfer_model.layers[:10]:
            layer.trainable = False
        for layer in transfer_model.layers[10:]:
            layer.trainable = True

        with open('/app/error_log.txt', "a") as log:
            log.write("Defining of model\n")

        # Compiling the model
        learning_rate = 1e-5 #change learning rate here
        optimizer = Adam(learning_rate)
        transfer_model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,  
            metrics=['accuracy']
        )


        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


        # Training the model
        try:
            epochs = 1 # Adjust the epochs as needed
            transfer_history = transfer_model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=validation_dataset,
                callbacks=[early_stopping]
            )
        except tf.errors.ResourceExhaustedError as e:
            print(f"Resource Exhausted Error during training: {e}")

        except Exception as e:
            print(f"An error occurred during model training: {e}")

        with open('/app/error_log.txt', "a") as log:
            log.write("Completed training of model\n")

        # Saving the trained model
        try:
            
            # Append the model file name to the save path
            full_model_save_path = os.path.join(model_save_path, 'trainedmodel.h5')
    
            # Save the model
            transfer_model.save(full_model_save_path)

            print(f"Model saved to {model_save_path}")
        except Exception as e:
            print(f"An error occurred while saving the model: {e}")

        # Extract the training history and final metrics
        training_metrics = {
            "accuracy": transfer_history.history['accuracy'][-1],
            "val_accuracy": transfer_history.history['val_accuracy'][-1],
            "loss": transfer_history.history['loss'][-1],
            "val_loss": transfer_history.history['val_loss'][-1],
        }

       # Construct the response to match the desired format
        response = {
            "accuracy": training_metrics["accuracy"],
            "val_accuracy": training_metrics["val_accuracy"],
            "loss": training_metrics["loss"],
            "val_loss": training_metrics["val_loss"],
            "model_params": {
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": 100, # Match the batch size used in training
                "optimizer": optimizer._class.name_
                }         
            }
        
        with open('/mnt/saved_model/output.json', 'w') as f:
            json.dump(response,f)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        with open('/app/error_log.txt', "a") as log:
            log.write(str(e) + "\n")


# DUMMY CODE, SHIFT THIS DECLARATION TO TOP WHEN LOCATION IS KNOWN
json_file_path = '../mnt/saved_model/output.json'

# get results
@app.route('/results', methods=['POST'])
def get_and_return_results():
    # read file from .js file stored in PV
    try:
        # Check if the file exists
        if not os.path.exists(json_file_path):
            return jsonify({"error": "File not found"}), 400
        
        # Open and read the JSON file
        with open('/mnt/saved_model/output.json', 'r') as json_file: # EDIT HERE
            data = json.load(json_file)
        
        # Return the data in the same JSON format
        return jsonify(data), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Running the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082) # same as deployment cfig