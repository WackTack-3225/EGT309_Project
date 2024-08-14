#!/bin/bash

# Handle minikube processess (assuming is downloaded)
minikube delete
minikube start
eval $(minikube docker-env)


# Run the deployment file
kubectl apply -f deployment.yml 

# Run container specific processes (if needed)

# build flask 
docker pull wacktack/flask-app:latest
# build data processing
docker pull jurnjie/data-processing:latest
# build model training
docker pull 200iqkid/model-training:latest
# build inference
docker pull edysontan/inference:latest


# Return the url of the website
minikube service flask-app-service -n ml-app --url