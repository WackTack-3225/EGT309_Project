#!/bin/bash


# Handle minikube processess (assuming is downloaded)
minikube delete
minikube start
eval $(minikube docker-env)


# Run the deployment file
kubectl apply -f deployment.yml 

# Run container specific processes (if needed)

# build docker containers (as they do not run for some reason)
cd minzheng/website









# Return the url of the website!