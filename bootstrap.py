from flask import Flask, request, jsonify
import tempfile
import os
import base64
from inference import get_model
import supervision as sv
from inference.core.utils.image_utils import load_image_bgr
import cv2
import numpy as np

app = Flask(__name__)

# Load the model once when the server starts
model = get_model(model_id="yolov8x-640")

@app.route('/analyze', methods=['POST'])
def analyze():
    # Check if an image file is in the request
    if 'image' not in request.files:
        return {"error": "No image provided"}, 400
    
    # Get the image from the request
    image_file = request.files['image']
    
    # Load the image as BGR from the uploaded file
    image = load_image_bgr(image_file)

    # Run inference
    results = model.infer(image)[0]
    detections = sv.Detections.from_inference(results)

    # Change all labels to "armor-vehicle"
    detections.labels = ["armor-vehicle"] * len(detections)

    # Annotate the image
    annotator = sv.BoxAnnotator(thickness=4)
    annotated_image = annotator.annotate(image, detections)
    annotator = sv.LabelAnnotator(text_scale=2, text_thickness=2)
    annotated_image = annotator.annotate(annotated_image, detections, labels=["armor-vehicle"] * len(detections))
    
    # Convert the annotated image to base64
    _, buffer = cv2.imencode('.jpg', annotated_image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    # Return the response as JSON
    response = {
        "annotated_image": image_base64,
        "detected_vehicles": len(detections)
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
