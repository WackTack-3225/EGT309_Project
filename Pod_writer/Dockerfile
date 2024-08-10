# Use an official Python runtime as a base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install TensorFlow or Keras, or any other dependencies needed for model saving
RUN pip install tensorflow keras

# Run your Python script
CMD ["python", "model_train.py"]