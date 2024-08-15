#!/bin/bash
minikube delete

minikube start --cpus=4 memory=4096

eval $(minikube docker-env) 

# Run the consolidated deployment file on start-up
kubectl apply -f main_deployment.yml

# Function to display the menu
function show_menu() {
    echo "1) Pod Overwatch"
    echo "2) Get pods status"
    echo "3) Get url to application"
    echo "4) Exit"
    echo -n "Please enter your choice: "
}

# Function to perform Task 1
function task1() {
    echo "The pods take a while to spin up, please have patience"
    echo "enter ctrl+c to exit overwatch mode"
    kubectl get pods -n ml-app -w
}

# Function to perform Task 2
function task2() {
    kubectl get pods -n ml-app
}

# Function to perform Task 3
function task3() {
    # Attempt to retrieve the service URL
    minikube service flask-app-service -n ml-app --url
}

# Main logic
while true
do
    show_menu
    read CHOICE
    case "$CHOICE" in
        1) task1 ;;
        2) task2 ;;
        3) task3 ;;
        4) echo "Exiting..."
           exit ;;
        *) echo "Invalid choice. Please select 1, 2, 3"
           echo "Press any key to continue..."
           read -n 1
    esac
done
