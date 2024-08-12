from flask import Flask, request, jsonify, render_template
import base64
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

### Inference START
@app.route('/inference', methods=['POST'])
def inference():
    # Check if any files were uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    files = request.files.getlist('image')  # Get list of uploaded files
    if len(files) == 0:
        return jsonify({'error': 'No selected files'}), 400

    results = []

    # Loop through each uploaded file
    for file in files:
        if file.filename == '':
            return jsonify({'error': 'One or more files have no filename'}), 400

        # Get the payload
        payload = request.form.get('payload')
        if not payload:
            return jsonify({'error': 'No payload provided'}), 400

        # Forward the image and payload to the data processing pod
        inference_result = forward_to_inference_pod(file, payload)

        # Check if the inference pod returned an error
        if 'error' in inference_result:
            return jsonify(inference_result), 400

        # Add the result to the list of results
        results.append(inference_result)

    # Return all inference results to the client
    return jsonify(results)

def forward_to_inference_pod(image_file, payload):
    INF_POD_URL = "http://inf-pod-service/process"  # Replace with pod URL

    files = {'image': image_file}
    data = {'payload': payload}

    try:
        response = requests.post(INF_POD_URL, files=files, data=data)
        if response.status_code == 200:  # expects status 200 with data
            response_data = response.json()

            # Assuming the inference pod returns prediction and confidence
            prediction = response_data.get('prediction')
            confidence = response_data.get('confidence')

            # Convert the image file to base64
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            image_url = f"data:image/png;base64,{image_base64}"

            return {
                "imageUrl": image_url,
                "prediction": prediction,
                "confidence": confidence
            }
        else:
            return {"error": "Data processing pod error", "status_code": response.status_code}
    
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to reach data processing pod: {e}"}
    
# Inference END
# INFERENCE SKELETON WORKING COMPLETELY, DO NOT MODIFY - 12/8/2024

def run_training_pipeline(payload): # sends code to training pipeline
    """
    This function forwards the payload to the data processing pod 
    and triggers the training pipeline.
    """
    DATA_PROCESSING_POD_URL = "http://data-processing-pod-service/process"  # Replace with the pod URL, need a flask for each pod and also find the cluster ip url

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
