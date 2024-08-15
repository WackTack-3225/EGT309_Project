#!/bin/bash

# Handle minikube processess (assuming is downloaded)
minikube delete
minikube start --cpus=4 memory=4096
eval $(minikube docker-env)

# Build ALL Docker images

# build flask 
docker pull wacktack/flask-app:latest
# build data processing
docker pull jurnjie/data-processing:latest
# build model training
docker pull 200iqkid/model-training:latest
# build inference
docker pull edysontan/inference:latest

# Run the consolidated deployment file
kubectl apply -f deployment.yml 

# Return the url of the website
minikube service flask-app-service -n ml-app --url