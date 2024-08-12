import os
from keras.applications import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
import tensorflow as tf

# Paths to the data
train_data_dir = os.getenv('TRAIN_DATA_PATH', '/mnt/data/train')
validation_data_dir = os.getenv('VALIDATION_DATA_PATH', '/mnt/data/validation')
model_save_path = os.getenv('MODEL_SAVE_PATH', '/mnt/models/my_model.h5')

# MobileNetV2 Model as base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation='relu')(x)
predictions = layers.Dense(4, activation='softmax')(x)
transfer_model = models.Model(inputs=base_model.input, outputs=predictions)

# Freezing base model layers
for layer in base_model.layers:
    layer.trainable = False
    print(layer.name, layer.trainable)

# # Data Augmentation
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )

# validation_datagen = ImageDataGenerator(rescale=1./255)

# train_augmented_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(64, 64),
#     batch_size=32,
#     class_mode='categorical'
# )

# validation_augmented_generator = validation_datagen.flow_from_directory(
#     validation_data_dir,
#     target_size=(64, 64),
#     batch_size=32,
#     class_mode='categorical'
# )

# Compile the model
transfer_model.compile(loss='categorical_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'])

# Train the model
transfer_history = transfer_model.fit(
    train_augmented_generator,
    epochs=2,  # Change this to 30 for full training
    validation_data=validation_augmented_generator
)

# Save the trained model
transfer_model.save(model_save_path)
print(f"Model saved to {model_save_path}")
