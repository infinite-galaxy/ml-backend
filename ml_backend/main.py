import base64
import cv2
import numpy as np
import logging
from ml_backend.application.detect import Detect
from ml_backend.domain.entities.rdd_features import RddFeatures
from ml_backend.domain.repositories.rdd_repository_abc import RddRepositoryABC
from ml_backend.infrastructure.repositories.yolo_rdd_repository import YoloRddRepository
from flask import Flask, jsonify, request


# Repositories
rdd_repository: RddRepositoryABC = YoloRddRepository()

# Use case
detect = Detect(rdd_repository)


def handler(job):
    """ Handler function that will be used to process jobs. """

    job_input = job['input']

    features = job_input.get('features')


app = Flask(__name__)
logging.basicConfig(level=logging.INFO)


@app.route('/detect', methods=['POST'])
def predict():
    data = request.get_json()

    if 'model_id' not in data:
        return jsonify({"error": "Missing 'model_id' key in JSON payload"}), 400
    if 'features' not in data:
        return jsonify({"error": "Missing 'features' key in JSON payload"}), 400

    model_id = data.get('model_id')
    features = data.get('features')

    match model_id:
        case 'rdd':
            features = RddFeatures(**features)

    try:
        # Decode Base64
        image_bytes = base64.b64decode(features.image)

        # Convert bytes to OpenCV image (BGR format)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image_bgr is None:
            return jsonify({"error": "Could not decode image data"}), 400

        features.image = image_bgr

    except base64.binascii.Error as b64_err:
        return jsonify({"error": f"Invalid base64 string: {b64_err}"}), 400
    except Exception as decode_err:
        return jsonify({"error": f"Error decoding image: {decode_err}"}), 400

    try:
        detections, image, times = detect(model_id, features)

        return jsonify({
            'status': 'success',
            'data': {
                'detections:': detections,
                'image': image,
                'meta': {
                    'model_id': model_id,
                    'times': times,
                }
            }
        })

    except Exception as e:
        # Log traceback
        app.logger.error(
            f"Error during prediction: {e}", exc_info=True
        )
        return jsonify({
            'status': 'error',
            'data': {
                'message': f"Prediction failed: {e}"
            }
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "success"}), 200


if __name__ == '__main__':
    print(f"Starting Flask server on port 8000...")
    # Use host='0.0.0.0' to make it accessible from outside the container/machine
    app.run(host='0.0.0.0', port=8000, debug=False)
