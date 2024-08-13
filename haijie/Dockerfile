# Pull in the processed data from Jurnjie's image
FROM jurnjie/data-processing:latest as data-stage

# Use your current base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the processed data from the first stage
COPY --from=data-stage /app/data_309 /app/data_309

# Copy the current directory contents into the container at /app
COPY . /app

# Install any dependencies needed for model training
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Run your training script
CMD ["python", "trainmodel.py"]