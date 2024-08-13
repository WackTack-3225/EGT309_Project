import os
import sys
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

try:
    # Define the paths to the training and validation data
    train_data_dir = '/app/data_309/train'
    validation_data_dir = '/app/data_309/test'

    # Ensure the directories exist
    if not os.path.exists(train_data_dir):
        raise FileNotFoundError(f"Training data directory not found: {train_data_dir}")
    if not os.path.exists(validation_data_dir):
        raise FileNotFoundError(f"Validation data directory not found: {validation_data_dir}")

    # Path to save the model
    model_save_path = os.getenv('MODEL_SAVE_PATH', '/mnt/saved_model/trained_model.h5')

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
    try:
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    except Exception as e:
        print(f"Error initializing MobileNetV2 base model: {e}")
        sys.exit(1)

    # Setting up the transfer learning model
    try:
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        predictions = layers.Dense(train_generator.num_classes, activation='softmax')(x)
        transfer_model = models.Model(inputs=base_model.input, outputs=predictions)
    except Exception as e:
        print(f"Error setting up the transfer model: {e}")
        sys.exit(1)

    # Freezing base model layers
    try:
        for layer in base_model.layers:
            layer.trainable = False
            print(f"{layer.name} trainable: {layer.trainable}")
    except Exception as e:
        print(f"Error freezing base model layers: {e}")
        sys.exit(1)

    # Compiling the model
    try:
        transfer_model.compile(loss='categorical_crossentropy',
                               optimizer='adam',
                               metrics=['accuracy'])
    except Exception as e:
        print(f"Error compiling the model: {e}")
        sys.exit(1)

    # Training the model
    try:
        transfer_history = transfer_model.fit(
            train_generator,
            epochs=30,  # Adjust the number of epochs as needed
            validation_data=validation_generator
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
