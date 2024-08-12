import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os

# # Define paths from environment variables
# train_df = os.getenv('TRAIN_DIR', '/app/data_309/train')
# test_df = os.getenv('TEST_DIR', '/app/data_309/test')

# try:
#     train_datagen = ImageDataGenerator(rescale=1./255)
#     test_datagen = ImageDataGenerator(rescale=1./255)

#     # Training data generator
#     train_generator = train_datagen.flow_from_directory(
#         train_df,
#         target_size=(64, 64),  # Resize 64x64
#         batch_size=100,
#         shuffle=True,
#         class_mode='categorical',
#     )

#     # Validation data generator
#     validation_generator = test_datagen.flow_from_directory(
#         test_df,
#         target_size=(64, 64),  # Resize 64x64
#         batch_size=100,
#         shuffle=True,
#         class_mode='categorical',
#     )

#     # Add your data processing logic here
#     print("Data generators created successfully")

# except Exception as e:
#     print(f"An error occurred during data processing: {e}")

print("help")