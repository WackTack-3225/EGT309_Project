# Use an official Python runtime as a base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Upgrades pip to ensure the latest library installations, then installs dependencies from requirements.txt for structured and organized dependency management.
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Runs Python script
CMD ["python", "predict.py"]