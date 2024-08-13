from flask import Flask, request, jsonify, render_template
import base64
import json
import subprocess
import requests
import os

app = Flask(__name__)

# Serve the HTML file
@app.route('/')
def index():
    return render_template('index.html')  

# Define routing for TRAIN
@app.route('/training_page')
def training_route():
    return render_template('training/01_1.html')  

# Define routing for INFERENCE
@app.route('/inference_page')
def inference_route():
    return render_template('inference/02_1.html')  

# Define routing for INFERENCE
@app.route('/inference_page_2')
def inference_route_2():
    return render_template('inference/02_2.html')  

# Endpoint to start the model training
@app.route('/process', methods=['POST'])
def start_training():
    try:
        # Send POST request to data processing pod to start the training
        DP_POD_URL = "http://data-processing-pod/start-training"  # Replace with the actual URL of your data processing pod
        payload = request.json

        response = requests.post(DP_POD_URL, json=payload)

        if response.status_code == 200 and response.json().get('success'):
            return jsonify({"success": True}), 200
        else:
            return jsonify({"success": False, "error": "Training pipeline failed"}), 500

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# Endpoint to get model accuracy and parameters
@app.route('/results', methods=['POST'])
def get_model_info():
    try:
        # Send POST request to inference pod to retrieve model data
        inference_pod_url = "http://inference-pod/get-model-info"  # Replace with the actual URL of your inference pod
        response = requests.post(inference_pod_url, json=request.json)

        if response.status_code == 200 and response.json().get('success'):
            data = response.json()
            return jsonify({
                "success": True,
                "accuracy": data.get('accuracy'),
                "parameters": data.get('parameters')
            }), 200
        else:
            return jsonify({"success": False, "error": "Failed to retrieve model data"}), 500

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

### Inference START
@app.route('/inference', methods=['POST'])
def inference():
    # Check if any files were uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    files = request.files.getlist('image')  # Get list of uploaded files
    if len(files) == 0:
        return jsonify({'error': 'No selected files'}), 400

    # Get the payload
    payload = request.form.get('payload')
    if not payload:
        return jsonify({'error': 'No payload provided'}), 400

    # Forward all images and the payload to the data processing pod
    inference_results = forward_to_inference_pod(files, payload)

    # Check if the inference pod returned an error
    if 'error' in inference_results:
        return jsonify(inference_results), 400

    # Return all inference results to the client
    return jsonify(inference_results)

def forward_to_inference_pod(files, payload):
    INF_POD_URL = "http://inference-service:80/predict"  # Replace with your pod's URL

    # Prepare the list of images in Base64 format
    images = []
    for file in files:
        image_base64 = base64.b64encode(file.read()).decode('utf-8')
        images.append({
            'image': image_base64,
        })
    
    # Create the JSON payload
    json_payload = {
        'images': images,
        'payload': payload
    }

    try:
        # Send the JSON data to the inference pod
        response = requests.post(INF_POD_URL, json=json_payload)
        
        if response.status_code == 200:  # Expecting status 200 with data
            return response.json()  # Assuming the response is a list of results
        else:
            return {"error": "Data processing pod error", "status_code": response.status_code}
    
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to reach data processing pod: {e}"}

    
# Inference END
# INFERENCE SKELETON WORKING COMPLETELY, DO NOT MODIFY - 12/8/2024


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
