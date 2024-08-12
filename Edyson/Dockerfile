# Use an official Python runtime as a base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install TensorFlow or Keras, or any other dependencies needed for model saving

# If you have dependencies in a requirements.txt file, uncomment the next line
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install tensorflow keras


# Run your Python script
CMD ["python", "retrieve.py"]