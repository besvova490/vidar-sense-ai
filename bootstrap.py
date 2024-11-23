from flask import Flask, request, send_file
import tempfile
import os
from inference import get_model
import supervision as sv
from inference.core.utils.image_utils import load_image_bgr
import cv2
import numpy as np

app = Flask(__name__)

# Load the model once when the server starts
model = get_model(model_id="yolov8s-640")

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
    results = sv.Detections.from_inference(results)

    # Annotate the image
    annotator = sv.BoxAnnotator(thickness=4)
    annotated_image = annotator.annotate(image, results)
    annotator = sv.LabelAnnotator(text_scale=2, text_thickness=2)
    annotated_image = annotator.annotate(annotated_image, results)
    
    # Save annotated image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        temp_filename = temp.name
        cv2.imwrite(temp_filename, annotated_image)

    # Send the file in the response
    return send_file(temp_filename, mimetype='image/jpeg', as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)

# Clean up temporary files after response
@app.after_request
def remove_temp_file(response):
    try:
        if response.direct_passthrough:
            return response
        temp_filename = response.headers.get("Content-Disposition").split("filename=")[-1].strip("\" ")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    except Exception as e:
        app.logger.error(f"Error deleting temporary file: {e}")
    return response
