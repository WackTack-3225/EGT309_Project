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

# Define routing for TRAIN_1
@app.route('/training_page')
def training_route():
    return render_template('training/01_1.html')  

# Define routing for INFERENCE_1
@app.route('/inference_page')
def inference_route():
    return render_template('inference/02_1.html')  

# Define routing for INFERENCE_2
@app.route('/inference_page_2')
def inference_route_2():
    return render_template('inference/02_2.html')  

# Endpoint to start the model training
@app.route('/process', methods=['POST'])
def start_training():
    try:
        # Send POST request to data processing pod to start the training
        DP_POD_URL = os.getenv("DP_POD_URL")

        response = requests.get(DP_POD_URL)

        if response.status_code == 200 and response.json() == {"status": "Processing completed"} :
            return jsonify({"success": True}), 200
        else:
            return jsonify({"success": False, "error": "Training pipeline failed"}), 500

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# Endpoint to get model accuracy and parameters
@app.route('/results', methods=['GET'])
def get_model_info():
    try:
        # Send POST request to inference pod to retrieve model data
        MODEL_TRAINING_URL = os.getenv("MODEL_TRAINING_URL")
        response = requests.post(MODEL_TRAINING_URL, json=request.json)

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
    if not request.files:
        return jsonify({'error': 'No files found in the request'}), 400
    
    # Prepare a list to hold image data in Base64
    images_data = []

    for key in request.files:
        file = request.files[key]

        # Read the file and encode it to Base64
        image_data = base64.b64encode(file.read()).decode('utf-8')

        # Append the Base64 encoded image data to the list
        images_data.append({'image': image_data})

    # Create a JSON payload
    json_payload = {
        'images': images_data,
        'payload': 200  # Add any additional data you need
    }

    # Forward all images and the payload to the data processing pod
    inference_results = forward_to_inference_pod(json_payload)

    # Check if the inference pod returned an error
    if 'error' in inference_results:
        return jsonify(inference_results), 400

    # Return all inference results to the client
    return jsonify(inference_results),200

def forward_to_inference_pod(json_payload):
    INF_POD_URL = os.getenv("INF_POD_URL")

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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
