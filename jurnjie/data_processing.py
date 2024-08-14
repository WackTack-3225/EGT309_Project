from flask import Flask, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import time
import numpy as np
from PIL import Image
from threading import Thread

app = Flask(__name__)

# Define paths
train_df = '/mnt/data/train'
test_df = '/mnt/data/test'
train_save_path = '/app/processed_data/train'
test_save_path = '/app/processed_data/test'

# Ensure the save directories exist
os.makedirs(train_save_path, exist_ok=True)
os.makedirs(test_save_path, exist_ok=True)

# Initialize ImageDataGenerators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Training data generator
train_generator = train_datagen.flow_from_directory(
    train_df,
    target_size=(64, 64),  # Resize 64x64
    batch_size=100,
    shuffle=True,
    class_mode='categorical',
)

# Validation data generator
validation_generator = test_datagen.flow_from_directory(
    test_df,
    target_size=(64, 64),  # Resize 64x64
    batch_size=100,
    shuffle=True,
    class_mode='categorical',
)

# Function to save images from generator
def save_images_from_generator(generator, save_path):
    total_batches = len(generator)
    for batch_index in range(total_batches):
        images, labels = next(generator)
        for i in range(images.shape[0]):
            image = (images[i] * 255).astype(np.uint8)
            img = Image.fromarray(image)
            label = np.argmax(labels[i])
            class_dir = os.path.join(save_path, str(label))
            os.makedirs(class_dir, exist_ok=True)
            img.save(os.path.join(class_dir, f'image_{batch_index * generator.batch_size + i}.png'))

# Route to start image processing
@app.route('/start-processing', methods=['GET'])
def start_processing():
    save_images_from_generator(train_generator, train_save_path)
    save_images_from_generator(validation_generator, test_save_path)
    return jsonify({"status": "Processing completed"}), 200

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "Service is running"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8084)