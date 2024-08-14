import os
import sys
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping


from flask import Flask, request, jsonify, render_template
import base64
import json
import subprocess
import requests

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train():
    # Retrieve the status code from the request headers
    status_code = request.headers.get('Status-Code', type=int)  # Typecast to int

    # Extract the payload from the request
    data = request.get_json()

    # Check if the JSON is present
    if not data:
        return jsonify({"error": "Missing or invalid JSON payload."}), 400
    
    value = data.get('payload')  # Extract the value from JSON, dummy code for model parameters

    # Check if the status code is 200
    if status_code != 200:
        return jsonify({"error": "Invalid status code. Expected 200."}), 400
    
    # main model training section
    try:
        # Define the paths to the training and validation data
        train_data_dir = '/app/data_309/train'
        validation_data_dir = '/app/data_309/test'

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

        # Path to save the model
        model_save_path = os.getenv('MODEL_SAVE_PATH', '/mnt/saved_model/trained_model.keras')

        if not model_save_path:
            raise ValueError("MODEL_SAVE_PATH environment variable is not set or is empty.")
        
        # Ensure the directory exists where the model will be saved
        model_save_dir = os.path.dirname(model_save_path)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        # Data generators for loading and augmenting the images
        try:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True
            )

            test_datagen = ImageDataGenerator(rescale=1./255)

            train_generator = train_datagen.flow_from_directory(
                train_data_dir,
                target_size=(64, 64),
                batch_size=32,
                class_mode='categorical'
            )

            validation_generator = test_datagen.flow_from_directory(
                validation_data_dir,
                target_size=(64, 64),
                batch_size=32,
                class_mode='categorical'
            )
        except Exception as e:
            print(f"Error in data loading or augmentation: {e}")
            sys.exit(1)


        # MobileNetV2 Model as base model
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(64, 64, 3))


        # Modify input shape and classifier layers
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        predictions = layers.Dense(train_generator.num_classes, activation='softmax')(x)
        transfer_model = models.Model(inputs=base_model.input, outputs=predictions)


        # Freezing the first 10 layers of the model
        base_model.trainable = True
        for layer in transfer_model.layers[:10]:
            layer.trainable = False
        for layer in transfer_model.layers[10:]:
            layer.trainable = True


        # Compiling the model
        transfer_model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(1e-5),  # Lower learning rate
            metrics=['accuracy']
        )


        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


        # Training the model
        try:
            transfer_history = transfer_model.fit(
                train_generator,
                epochs=2,  # Adjust the number of epochs as needed
                validation_data=validation_generator,
                callbacks=[early_stopping]
            )
        except tf.errors.ResourceExhaustedError as e:
            print(f"Resource Exhausted Error during training: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"An error occurred during model training: {e}")
            sys.exit(1)

        # Saving the trained model
        try:
            transfer_model.save(model_save_path)
            print(f"Model saved to {model_save_path}")
        except Exception as e:
            print(f"An error occurred while saving the model: {e}")
            sys.exit(1)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

    print("trainmodel.py execution completed.")

    # Sending a response back to the server with a status code of 200
    try:
        SERVER_URL = 'https//dummy.com'
        response = requests.post(
            SERVER_URL,  # Replace with the actual URL and endpoint
            json={"payload": 101} # dummy code for model parameters & accuracy
        )
        print(f"POST request sent to the server. Response status: {response.status_code}")
    except Exception as e:
        print(f"An error occurred while sending the POST request: {e}")
        sys.exit(1)

    return jsonify(response.json()), 200


# Running the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082) # -2 for 2nd process in pipeline