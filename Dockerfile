# Use the official Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Set environment variables to prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Run the Python script when the container launches
CMD ["python", "data_processing.py"]
