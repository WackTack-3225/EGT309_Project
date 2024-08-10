import tensorflow as tf
from tensorflow.keras import layers, models

# Create a simple model for demonstration purposes
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(100,)),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save the model to the shared volume
model.save('/mnt/data/model_name.h5')

print("Model saved to /mnt/data/model_name.h5")