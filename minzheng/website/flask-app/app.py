from flask import Flask, request, jsonify, render_template
import subprocess
import requests
import os

app = Flask(__name__)

# Serve the HTML file
@app.route('/')
def index():
    return render_template('index.html')  

# Serve the pipeline progress HTML file
@app.route('/training')
def training():
    return render_template('training/01_1.html')  # Serve the 01_1.html file from the training directory

# Pipeline Code START
# Endpoint for pods to send their status updates
@app.route('/update_status', methods=['POST'])  
def update_status():
    global current_step, inference_result
    data = request.get_json()
    current_step = data.get('step')

    # Check if the inference pod sent back results
    if 'inference_result' in data:
        inference_result = data['inference_result']

    return jsonify({"status": "updated", "current_step": current_step})

# Endpoint for the frontend to check the current progress
@app.route('/progress', methods=['GET'])
def progress():
    global inference_result
    response = {"step": current_step, "inference_result": inference_result}

    # Clear the varaibles after returning it to the client
    inference_result = None
    current_step = None

    return jsonify(response)

# Pipeline Code END

@app.route('/process', methods=['POST'])
def process_payload():
    data = request.get_json()
    if data['message'] == 'run_training_pipeline':
        result = run_training_pipeline(data['payload'])
    else:
        result = {"error": "Invalid pipeline"}
    return jsonify(result)

# Inference START

@app.route('/inference', methods=['POST'])
def inference():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Get the payload
    payload = request.form.get('payload')
    if not payload:
        return jsonify({'error': 'No payload provided'}), 400

    # Forward the image and payload to the data processing pod
    processed_image_data = forward_to_data_processing_pod(file, payload)

    # Check if the data processing pod returned an error
    if 'error' in processed_image_data:
        return jsonify(processed_image_data), 400

    # Now forward the processed image data to the inference pod
    inference_result = forward_to_inference_pod(processed_image_data)

    # Return the final inference result to the client
    return jsonify(inference_result)

def forward_to_data_processing_pod(image_file, payload):
    """
    This function sends the image and payload to the data processing pod
    and returns the processed image data.
    """
    DATA_PROCESSING_POD_URL = "http://data-processing-pod-service/process"  # Replace with pod URL

    files = {'image': image_file}
    data = {'payload': payload}

    try:
        response = requests.post(DATA_PROCESSING_POD_URL, files=files, data=data)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "Data processing pod error", "status_code": response.status_code}
    
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to reach data processing pod: {e}"}

def forward_to_inference_pod(processed_image_data):
    """
    This function sends the processed image data to the inference pod
    and returns the inference result.
    """
    INFERENCE_POD_URL = "http://inference-pod-service/inference"  # Replace with pod URL

    try:
        response = requests.post(INFERENCE_POD_URL, json=processed_image_data)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "Inference pod error", "status_code": response.status_code}
    
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to reach inference pod: {e}"}
    
# Inference END

def run_training_pipeline(payload): # sends code to training pipeline
    """
    This function forwards the payload to the data processing pod 
    and triggers the training pipeline.
    """
    DATA_PROCESSING_POD_URL = "http://data-processing-pod-service/process"  # Replace with the pod URL

    data = {'payload': payload}

    try:
        response = requests.post(DATA_PROCESSING_POD_URL, json=data)
        if response.status_code == 100:
            return response.json()
        else:
            return {"error": "Data processing pod error", "status_code": response.status_code}
    
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to reach data processing pod: {e}"}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
