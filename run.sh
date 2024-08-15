#!/bin/bash

# Handle minikube processess (assuming is downloaded)
minikube delete
minikube start --cpus=4 memory=4096
eval $(minikube docker-env) 

# Run the consolidated deployment file on start-up
kubectl apply -f deployment.yml

# Function to display the menu
function show_menu() {
    echo "1) Get pods status"
    echo "2) Get url to application"
    echo "3) Exit"
    echo -n "Please enter your choice: "
}

# Function to perform Task 1
function task1() {
    kubectl get pods -n ml-app -w
}

# Function to perform Task 2
function task2() {
    # Attempt to retrieve the service URL
    SERVICE_URL=$(minikube service flask-app-service -n ml-app --url)
    
    if [[ $SERVICE_URL =~ http:// ]]; then
        echo "Service URL: $SERVICE_URL"
    else
        echo "No valid URL returned. Please try again."
    fi
}

# Main logic
while true
do
    show_menu
    read CHOICE
    case "$CHOICE" in
        1) task1 ;;
        2) task2 ;;
        3) echo "Exiting..."
           exit ;;
        *) echo "Invalid choice. Please select 1, 2, 3"
           echo "Press any key to continue..."
           read -n 1
    esac
done
