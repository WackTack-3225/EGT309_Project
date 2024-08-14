#!/bin/bash

# Handle minikube processess (assuming is downloaded)
minikube delete # reset environment
minikube start # begin environment
eval $(minikube docker-env) # enable docker commands


# Run the deployment file
kubectl apply -f deployment.yml 

# Run container specific processes

# build flask 
docker pull wacktack/flask-app
# build data processing
docker pull jurnjie/data-processing
# build model training
docker pull 200iqkid/model-training
# build inference
docker pull edysontan/inference


# Return the url of the website
minikube service flask-app-service -n ml-app --url