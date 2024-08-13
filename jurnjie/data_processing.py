import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import time
import numpy as np
from PIL import Image

# Define paths from environment variables
train_df = '/mnt/data/train'
test_df = '/mnt/data/test'
train_save_path = '/app/data_309/train_gen'
test_save_path = '/app/data_309/test_gen'

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
        images, labels = next(generator)  # Use next(generator) instead of generator.next()
        for i in range(images.shape[0]):
            image = (images[i] * 255).astype(np.uint8)  # Convert image back to 0-255 range
            img = Image.fromarray(image)
            label = np.argmax(labels[i])  # Assuming one-hot encoding, get the class label
            class_dir = os.path.join(save_path, str(label))
            os.makedirs(class_dir, exist_ok=True)
            img.save(os.path.join(class_dir, f'image_{batch_index * generator.batch_size + i}.png'))

# Save images from the train_generator
save_images_from_generator(train_generator, train_save_path)

# Save images from the validation_generator
save_images_from_generator(validation_generator, test_save_path)

while True:
    print("Images saved successfully.")
    time.sleep(100)


