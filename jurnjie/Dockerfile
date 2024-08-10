# Use an official Python runtime as a base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Copy the contents of the toinput folder into /mnt/newdata inside the container
COPY data_309 /mnt/data

# If you have dependencies in a requirements.txt file, uncomment the next line
# RUN pip install --no-cache-dir -r requirements.txt

# Run your Python script
CMD ["python", "data_prep.py"]