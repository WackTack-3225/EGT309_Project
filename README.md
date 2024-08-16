# AI Solution Architect Project 

**Goal: Deploy an end to end machine learning application in Kubernetes**

This is a group project where we deployed a 3 section machine learning pipeline on Kubernetes. We then further added a web page to allow for user interaction and communication with the pipeline over a GUI. 

This project aims to apply our skills learnt in class, as well as the various methodologies and attributes in a modern application and microservice architecture. We were also tasked to manage and handle various aspects of deployment, scaling, and monitoring, ensuring a robust and scalable solution.

# Project Architecture

![Project Architecture and connections](/.images_readme/architecture.png)

## Pipeline 

![Project Pipelines](/.images_readme/pipeline.png)

### **Model Training Pipeline**
The Model Training Pipeline involves the web server sending a POST request to the Data Processing Pod. The Data Processing Pod then sends back a response after the processing is complete. The Data Processing pod stores the processed data in Persistent Volume 1. 

After the web server receives a good response from the Data Processing Pod, it will then make a 2nd POST request to the Model Training Pod. The Model Training Pod then begins to take the processed data from Persistent Volume 1, and trains a deep learning model. It then writes and saves the model parameters and outputs into a json file into Persistent Volume 2. it also saves the trained model. The Model Training Pod then returns a response back to the web server.

Shortly after, the web server will send a 3rd POST request back to the Model Training Pod to retrieve the model results and parameters in a JSON Message. It will then display the results on the web page.

### **Inference Pipeline**
A user will choose to upload images to send for inference via the upload button. When the button is clicked to start the inference, the uploaded images are then processed into Base64 JSON encodings, where they will then by sent to the Inference Pod over a JSON Message.

The web server will make a POST request to the inference pod, where it will decode the Message sent, retrieve a saved model and process the images. The images are then returned back in Base64 JSON encodings alongside the predicted label and confidence in a JSON Message in the response. 

The web server will then display the images and results on the webpage after unpacking the images from the Base64 encoding.

# Project Components

## 1. **Web Interface**

> Contributor: Tan Min Zheng

- Description: The web interface provides a graphical user interface (GUI) for interacting with the machine learning pipeline. Users can upload data, request predictions, and view the results directly from the web page. The web service communicates with the other services in the pipeline through RESTful APIs. The images uploaded are sent to the inference section via JSON messages in Base64 encoding, where the inference pod then decrypts the images and processes them. 

## 2. **Data Processing**

> Contributor: Lee Jurn Jie

- Description: This section handles the preprocessing of raw data to prepare it for model training. This includes cleaning the data, feature engineering, and splitting the data into training and validation sets.

## 3. **Model Training**

> Contributor: Lim Hai Jie

- Description: The model training section is responsible for training a machine learning model using the processed dataset. This includes model selection, and training with various hyperparameters. The trained model as is then saved to a persistent storage location in the Kubernetes cluster. The hyperparameters is also saved in a JSON file for the flask-app to access and display in the Web Interface.

## 4. **Model Prediction**

> Contributor: Edyson Tan Guan Teck

- Description: The prediction section loads the trained model and uses it to make predictions on new data. It involves receiving base64-encoded images from a JSON payload from the web app through a POST request, converting the tensors from the ML model that is loaded from model-training to predict and lastly, prepare and return the results as JSON responses for display on a web interface. This is exposed as a service within the Kubernetes cluster, allowing other services or the web interface to request predictions.

> [!NOTE]
> Unfortunately, the inference code was not optimized to handle PNG images due to them having 4 color channels (RGBA). The current inference pipeline only processes JPEG/JPG properly with 3 color channels (RGB). 

## 5. Kubernetes Deployment

Each section of the project is containerized using Docker and deployed on a Kubernetes cluster. The deployment leverages Kubernetes' capabilities for scaling, service discovery, and fault tolerance. The overall architecture ensures that the solution is highly available, scalable, and can handle varying workloads.

Key Components:

1. Pods: 
- Each section runs in its own set of pods, ensuring isolation and scalability. [*They are deployed under `Kind: Deployment` for convenience and consistency between sections*]

2. Services: 
- Kubernetes Services are used to expose each section of the pipeline, making it accessible within the cluster. NodePorts and ClusterIP's are used for connection. 

3. Persistent Volumes: 
- Used to store the trained model and intermediate data, ensuring that data persists even if pods are restarted.

4. ConfigMaps: 
- Manage configuration data separately from the application code, allowing for easy updates and management. Especially for indicating filepaths and endpoints

# File Structure

└── EGT309_Project

    ├── Edyson

    │   ├── Dockerfile                                  # file to build docker images for inference

    │   ├── deployment.yaml                             # deployment configurations file for inference section

    │   ├── predict.py                                  # model training code & flask app for RESTful API

    │   └── requirements.txt                            # python requirements for inference section

    ├── haijie

    │   ├── Dockerfile                                  # file to build docker images for model training

    │   ├── deployment.yml                              # deployment configuration file for model training section

    │   ├── requirements.txt                            # python requirements for model training

    │   └── trainmodel.py                               # model training code & flask app for RESTful API

    ├── jurnjie

    │   ├── Dockerfile                                  # file to build docker images for data preprocessing

    │   ├── data_processing.py                          # model training code & flask app for RESTful API

    │   ├── deployment.yaml                             # deployment configuration file for data preprocessing section

    │   └── requirements.txt                            # python requirements for data processing

    ├── minzheng

    │   ├── deploy.yml

    │   └── website

    │       ├── Dockerfile                              # file to build docker images for web page

    │       └── flask-app

    │           ├── app.py                              # flask app for RESTful API

    │           ├── requirements.txt                    # python requirements for the webpage

    │           └── templates

    │               ├── index.html                      # Page to select pipelines

    │               ├── inference

    │               │   ├── 02_1.html                   # Page to trigger model training pipeline

    │               │   └── 02_2.html                   # Page to display results

    │               └── training

    │                   ├── 01_1.html                   # Page to trigger inference pipeline

    │                   └── 01_2.html                   # Page to display inference

    ├── README.md                                       # readme file containing project details

    ├── main_deployment.yml                                  # application deployment file

    └── run.sh                                          # shell script for deployment and access of the application

# How to run the application

This application is only capable of running in the local host of your computer. 

Application requirements:

- WSL (Windows Subsystem for Linux) [or a Linux OS]

- Docker

- Minikube 

## Steps:

1. **Download the folder from this GitHub repository**  
> Note: if you are using WSL, save the folder within the WSL directory so they are accessible

2. **Enter into the folder [EGT309_Project] in the location that you downloaded the repository in** 
> Run the following code: 
>
>```bash
> sed -i 's/\r$//' run.sh # removes syntax errors when pulling this file from GitHub, safety measure
>
> bash run.sh # runs the .sh file to start up pods by deploying the main_deployment.yml & get the application url by selecting an option
> ```
Assuming you have all the required systems in place (docker, minikube), the `bash run.sh` will automatically set up the minikube environment for you and deploy the application automatically.

3. **After starting up successfully you will be presented with a Menu**

> **Option 1: Pod Overwatch**
>   - This option monitors the status of the pods in real-time. It uses `kubectl get pods -n ml-app -w`, which continuously watches the status of the pods until you exit the mode using Ctrl+C.
>
> **Option 2: Get Pods Status**
>   - This option provides a snapshot of the current status of all pods in the ml-app namespace. It runs `kubectl get pods -n ml-app`.
>
> **Option 3: Get Application URL**
>   - This option retrieves the URL to access the Flask application deployed in the ml-app namespace using `minikube service flask-app-service -n ml-app --url`. This URL can be used in 
your browser to interact with the web application.
>   - If you experience any errors when retriving the URL, it is due to the flask-app pod not being spun up yet or not finished deploying. Check with Option 1 or 2
>
> **Option 4: Exit**
>   - Exits the menu and stops the script execution.

> [!WARNING]
> When you exit the script, you need to re-run the script. If you exit the application url, you need to enter the option to get the application url again to get a working url. You cannot exit the application url in the terminal and continue using the application as the Minikube Tunnel may assign a new URL to access the website.
