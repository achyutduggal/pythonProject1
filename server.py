from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# Hugging Face API endpoint and headers
API_URL = "https://api-inference.huggingface.co/models/nateraw/food"
headers = {"Authorization": "Bearer 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"}

def query(image_bytes):
    response = requests.post(API_URL, headers=headers, data=image_bytes)
    return response.json()

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    # Read the image file directly without saving
    image_bytes = file.read()

    # Use the Hugging Face Inference API to get predictions
    output = query(image_bytes)

    # Check for errors in the response
    if 'error' in output:
        return jsonify({'error': output['error']}), 500

    # Extract the predicted class and confidence
    if isinstance(output, list) and len(output) > 0:
        predicted_class = output[0]['label']
        confidence = output[0]['score']
    else:
        return jsonify({'error': 'Invalid response from model'}), 500

    # Return the result as JSON
    return jsonify({'class': predicted_class, 'confidence': float(confidence)})

if __name__ == '__main__':
    app.run(debug=True)
