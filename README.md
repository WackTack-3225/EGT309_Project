# AI Solution Architect Project 

**Goal: Deploy an end to end machine learning application in Kubernetes**

This is a group project where we deployed a 3 section machine learning pipeline on Kubernetes. We then further added a web page to allow for user interaction and communication with the pipeline over a GUI. 

This project aims to apply our skills learnt in class, as well as the various methodologies and attributes in a modern application and microservice architecture. We were also tasked to manage and handle We were also tasked to manage and handle various aspects of deployment, scaling, and monitoring, ensuring a robust and scalable solution.

# Project Sections

Our application consists of 4 key sections:

1. Web Interface

- Contributor: Tan Min Zheng

- Description: The web interface provides a graphical user interface (GUI) for interacting with the machine learning pipeline. Users can upload data, request predictions, and view the results directly from the web page. The web service communicates with the other services in the pipeline through RESTful APIs.

2. Data Processing

- Contributor: Lee Jurn Jie

- Description: This section handles the preprocessing of raw data to prepare it for model training. This includes cleaning the data, feature engineering, and splitting the data into training and validation sets.

3. Model Training

- Contributor: Lim Hai Jie

- Description: The model training section is responsible for training a machine learning model using the processed dataset. This includes model selection, and training with various hyperparameters. The trained model as is then saved to a persistent storage location in the Kubernetes cluster. The hyperparameters is also saved in a JSON file for the flask-app to access and display in the Web Interface.

4. Model Prediction

- Contributor: Edyson Tan Guan Teck

- Description: The prediction section loads the trained model and uses it to make predictions on new data. This is exposed as a service within the Kubernetes cluster, allowing other services or the web interface to request predictions.

# Kubernetes Deployment

Each section of the project is containerized using Docker and deployed on a Kubernetes cluster. The deployment leverages Kubernetes' capabilities for scaling, service discovery, and fault tolerance. The overall architecture ensures that the solution is highly available, scalable, and can handle varying workloads.

Key Components:

1. Pods: Each section runs in its own set of pods, ensuring isolation and scalability.

2. Services: Kubernetes Services are used to expose each section of the pipeline, making it accessible within the cluster.

3. Persistent Volumes: Used to store the trained model and intermediate data, ensuring that data persists even if pods are restarted.

4. ConfigMaps: Manage configuration data separately from the application code, allowing for easy updates and management.

# File Structure

└── EGT309_Project

    ├── Edyson

    │   ├── Dockerfile

    │   ├── Toinput

    │   │   ├── image.png

    │   │   └── model_name.h5

    │   ├── deployment.yaml

    │   ├── predict.py

    │   └── requirements.txt

    ├── haijie

    │   ├── Dockerfile

    │   ├── deployment.yml

    │   ├── requirements.txt

    │   └── trainmodel.py

    ├── jurnjie

    │   ├── Dockerfile

    │   ├── data_processing.py

    │   ├── deployment.yaml

    │   └── requirements.txt

    ├── minzheng

    │   ├── deploy.yml

    │   └── website

    │       ├── Dockerfile

    │       └── flask-app

    │           ├── app.py

    │           ├── requirements.txt

    │           └── templates

    │               ├── index.html

    │               ├── inference

    │               │   ├── 02_1.html

    │               │   └── 02_2.html

    │               └── training

    │                   ├── 01_1.html

    │                   └── 01_2.html

    ├── README.md

    ├── deployment.yml

    └── run.sh

# How to run the application

Unfortunately, this application has not been rehosted to a cloud server. This application however is capable of running in the local host of your computer!

Application requirements:

- WSL (Windows Subsystem for Linux) [or a Linux OS]

- Docker

- Minikube (Kubernetes)



1. Download the folder from github.  \
Note: if you are using WSL, save the folder within the WSL directory so they are accessible
2. enter into the folder (cd EGT309_Project) and run the following code: \
sed -i 's/\r$//' run.sh # removes syntax errors due when pulling this file from github \
Bash run.sh # runs the .sh file to start up pods & get the application url
3. Assuming you have all the required systems in place (docker, minikube), the bash run.sh will automatically set up the minikube environment for you 
4. After starting up successfully you will be presented with a UI \
 \


<p id="gdcalert1" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image1.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert2">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image1.png "image_tooltip")
 \
1) Pod Overwatch \
It runs the command: kubectl get pods -n ml-apps -w \
It updates the status of the pod, allowing you to check if the pods are spinning up properly \
2) Get pod status \
 \
 \
(FK I LAZY HELP ME TYPE) \
HERE IS THE WHOLE SH FILE
5. `#!/bin/bash`
6. `minikube delete`
7. 
8. `minikube start --cpus=4 memory=4096`
9. 
10. `eval $(minikube docker-env)`
11. 
12. `# Run the consolidated deployment file on start-up`
13. `kubectl apply -f main_deployment.yml`
14. 
15. `# Function to display the menu`
16. `function show_menu() {`
17. `    echo "1) Pod Overwatch"`
18. `    echo "2) Get pods status"`
19. `    echo "3) Get url to application"`
20. `    echo "4) Exi`
21. `    echo -n "Please enter your choice: "`
22. `}`
23. 
24. `# Function to perform Task 1`
25. `function task1() {`
26. `    echo "The pods take a while to spin up, please have patience"`
27. `    echo "enter ctrl+c to exit overwatch mode"`
28. `    kubectl get pods -n ml-app -w`
29. `}`
30. 
31. `# Function to perform Task 2`
32. `function task2() {`
33. `    kubectl get pods -n ml-app`
34. `}`
35. 
36. `# Function to perform Task 3`
37. `function task3() {`
38. `    # Attempt to retrieve the service URL`
39. `    minikube service flask-app-service -n ml-app --url`
40. `}`
41. 
42. `# Main logic`
43. `while true`
44. `do`
45. `    show_menu`
46. `    read CHOICE`
47. `    case "$CHOICE" in`
48. `        1) task1 ;;`
49. `        2) task2 ;;`
50. `        3) task3 ;;`
51. `        4) echo "Exiting..."`
52. `           exit ;;`
53. `        *) echo "Invalid choice. Please select 1, 2, 3"`
54. `           echo "Press any key to continue..."`
55. `           read -n 1`
56. `    esac`
57. `done`
58. 

First, navigate to the root directory of the project using windows powershell and type ‘wsl’

**Cluster Initialization:**

minikube delete: This command deletes any existing Minikube cluster to ensure a clean environment.

minikube start --cpus=4 --memory=4096: Starts a new Minikube cluster with 4 CPU cores and 4096 MB of memory. This provides sufficient resources for running the project.

eval $(minikube docker-env): Configures your shell to use Minikube's Docker daemon, allowing you to build Docker images directly within the Minikube environment.

**Deployment Setup:**

kubectl apply -f main_deployment.yml: This command applies the Kubernetes deployment configuration (main_deployment.yml), setting up the entire environment (e.g., pods, services) as defined in the YAML file.

Menu System:

The script includes a simple menu-driven interface that allows users to interact with the Kubernetes environment using the following options:

1) Pod Overwatch:

- This option monitors the status of the pods in real-time. It uses kubectl get pods -n ml-app -w, which continuously watches the status of the pods until you exit the mode using Ctrl+C.

2) Get Pods Status:

- This option provides a snapshot of the current status of all pods in the ml-app namespace. It runs kubectl get pods -n ml-app.

3) Get URL to Application:

- This option retrieves the URL to access the Flask application deployed in the ml-app namespace using minikube service flask-app-service -n ml-app --url. This URL can be used in your browser to interact with the web application.

4) Exit:

- Exits the menu and stops the script execution.

**Main Loop:**

The script runs in a continuous loop, displaying the menu until the user chooses to exit by selecting option 4.