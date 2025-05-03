import base64
import time
from typing import Tuple
import cv2
import numpy as np
from onnxruntime import InferenceSession
from os import path
from math import exp
from ml_backend.domain.entities.rdd_features import RddFeatures
from ml_backend.domain.repositories.rdd_repository_abc import RddRepositoryABC
from ml_backend.infrastructure.dtos.detect_box import DetectBox


class YoloRddRepository(RddRepositoryABC):
    CLASSES = [
        'LONGITUDINAL_CRACK',
        'TRANSVERSE_CRACK',
        'ALLIGATOR_CRACK',
        'POTHOLES',
    ]
    COLORS = [
        [255, 142, 0],
        [197, 68, 205],
        [228, 66, 52],
        [37, 210, 209],
    ]
    # Default thresholds
    DEFAULT_CONF_THRESH = 0.05
    DEFAULT_IOU_THRESH = 0.45
    # Default input size
    DEFAULT_INPUT_WIDTH = 640
    DEFAULT_INPUT_HEIGHT = 640

    def __init__(self):
        """Initializes the YOLO Road Damage Detector Repository."""
        current_dir = path.dirname(path.abspath(__file__))
        model_filename = 'rdd.onnx'
        model_path = path.join(current_dir, '..', 'resources', model_filename)

        if path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = f.read()
        else:
            raise FileNotFoundError(
                f"Model file '{model_filename}' not found in expected locations.")

        self.session = InferenceSession(
            model, providers=['CPUExecutionProvider']
        )

        # Dapatkan metadata input/output dari model
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        model_shape = input_meta.shape

        try:
            self.input_height = int(model_shape[2]) if isinstance(
                model_shape[2], int) else self.DEFAULT_INPUT_HEIGHT
            self.input_width = int(model_shape[3]) if isinstance(
                model_shape[3], int) else self.DEFAULT_INPUT_WIDTH
        except (TypeError, IndexError):
            print(
                f"Warning: Could not determine input size from model shape {model_shape}. Using defaults height ({self.DEFAULT_INPUT_HEIGHT}) and width ({self.DEFAULT_INPUT_WIDTH})"
            )
            self.input_height = self.DEFAULT_INPUT_HEIGHT
            self.input_width = self.DEFAULT_INPUT_WIDTH

        # Dapatkan nama output (mungkin ada lebih dari satu)
        self.output_names = [self.session.get_outputs()[0].name]

        self.class_num = len(self.CLASSES)

    def _preprocess(self, image_bgr: np.ndarray) -> tuple[np.ndarray, int, int]:
        """
        Prepares the image for YOLO inference (resize, RGB, normalize, CHW).

        Args:
            image_bgr (np.ndarray): Input image in BGR format.

        Returns:
            out (np.ndarray, (int, int), (int, int)):
                - input_tensor: The preprocessed image tensor (NCHW, float32).
                - (original_height, original_width): The original height and width of the image.
                - (top, left): The top and left padding value.
        """
        original_height, original_width = image_bgr.shape[:2]

        # Resize image while maintaining aspect ratio with padding
        ratio = min(
            self.input_width / original_width,
            self.input_height / original_height,
        )
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)

        resized_image = cv2.resize(
            image_bgr,
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR,
        )

        # Use a neutral color for padding (commonly gray for YOLO)
        color = [114, 114, 114]

        # Create a new image with padding
        delta_w = (self.input_width - new_width) / 2
        delta_h = (self.input_height - new_height) / 2
        top, bottom = int(round(delta_h - 0.1)), int(round(delta_h + 0.1))
        left, right = int(round(delta_w - 0.1)), int(round(delta_w + 0.1))

        padded_image = cv2.copyMakeBorder(
            resized_image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=color,
        )

        # Konversi BGR ke RGB
        image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
        # Normalisasi (0-1)
        image = image.astype(np.float32) / 255.0
        # Transpose HWC ke CHW
        image = image.transpose((2, 0, 1))
        # Tambah dimensi batch (NCHW)
        input_tensor = np.expand_dims(image, axis=0)

        return input_tensor, (original_height, original_width), (top, left)

    def _postprocess(self, outputs: list[np.ndarray], original_size: Tuple[int, int], padding: Tuple[int, int]) -> list[DetectBox]:
        """
        Decodes the raw model output tensors into bounding boxes, scores, and class IDs.

        Args:
            outputs (List[np.ndarray]): List of arrays from the model's output tensors.
            original_size (Tuple[int, int]): The original height and width of the image.
            padding (Tuple[int, int]): The top and left padding value applied during preprocessing.

        Returns:
            out (List[DetectBox]): A list of DetectBox objects after NMS.
        """
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(outputs[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Get the original image dimensions
        original_height, original_width = original_size

        # Calculate the scaling factors for the bounding box coordinates
        ratio = min(
            self.input_width / original_width,
            self.input_height / original_height,
        )
        outputs[:, 0] -= padding[1]
        outputs[:, 1] -= padding[0]

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = float(np.amax(classes_scores))

            # If the maximum score is above the confidence threshold
            if max_score >= self.DEFAULT_CONF_THRESH:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                width = int(w / ratio)
                height = int(h / ratio)
                left = int((x - w / 2) / ratio)
                top = int((y - h / 2) / ratio)
                right = left + width
                bottom = top + height

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, right, bottom])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(
            boxes, scores, self.DEFAULT_CONF_THRESH, self.DEFAULT_IOU_THRESH,
        )

        # Initialize the detection results list
        detect_results = []

        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Buat objek DetectBox
            box = DetectBox(
                class_ids[i],
                round(scores[i], 4),
                boxes[i][0],
                boxes[i][1],
                boxes[i][2],
                boxes[i][3],
            )
            detect_results.append(box)

        return detect_results

    def _draw_detections(self, image: np.ndarray, detections: list[DetectBox], mask_alpha=0.3) -> np.ndarray:
        """
        Draws bounding boxes, labels, and scores on the image.

        Args:
            image (np.ndarray): Input image (BGR format).
            detections (list[DetectBox]): List of DetectBox objects.
            mask_alpha: Transparency for the mask overlay.

        Returns:
            Image with detections drawn (BGR format).
        """
        det_img = image.copy()
        img_height, img_width = image.shape[:2]
        font_size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        # Buat gambar mask terpisah (opsional, bisa juga digambar langsung)
        mask_img = det_img.copy()

        for detection in detections:
            # Pastikan class_id valid
            if detection.class_id < 0 or detection.class_id >= len(self.CLASSES):
                print(
                    f"Warning: Invalid class ID {detection.class_id} encountered during drawing."
                )
                continue

            color = self.COLORS[detection.class_id]
            x1, y1, x2, y2 = detection.xmin, detection.ymin, detection.xmax, detection.ymax

            # Draw mask
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

            # Draw bounding box
            cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)

            # Prepare caption text
            label = self.CLASSES[detection.class_id]
            caption = f'{label} {int(detection.score * 100)}%'

            # Calculate text size
            (tw, th), _ = cv2.getTextSize(
                text=caption,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_size,
                thickness=text_thickness,
            )
            th = int(th * 1.5)

            # Draw background rectangle for text label
            cv2.rectangle(
                det_img,
                (x1, y1),
                (x1 + tw, y1 + th),
                color, -1,
            )

            # Draw text label
            cv2.putText(
                # Adjusted slightly for better centering
                det_img, caption,
                (x1, y1 + th - (th // 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )

        # Apply mask overlay if alpha is set
        if mask_alpha > 0:
            det_img = cv2.addWeighted(
                mask_img, mask_alpha, det_img, 1 - mask_alpha, 0,
            )

        return det_img

    def detect(self, features: RddFeatures) -> Tuple[list, str, dict]:
        """
        Detects objects using the YOLO model and returns detailed results.

        Args:
            features (RddFeatures): Features object containing the input image
                under the 'image' key (as a numpy BGR ndarray).

        Returns:
            out (List, str, dict):
                - detections: A list of list, each with the format [class_id, score, xmin, ymin, xmax, ymax].
                - image: Base64 encoded string of the image with detections drawn.
                - times: A dictionary with 'preprocess_time', 'inference_time', and 'postprocess_time' in seconds.
        """

        # 1. Preprocessing
        start_pre_time = time.time()
        image_bgr = features.image
        input_tensor, original_size, padding = self._preprocess(image_bgr)
        preprocess_time = time.time() - start_pre_time

        # 2. Inference
        start_inf_time = time.time()
        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_tensor}
        )
        inference_time = time.time() - start_inf_time

        # 3. Postprocessing
        start_post_time = time.time()
        detections = self._postprocess(outputs, original_size, padding)

        results = []
        # Convert DetectBox objects to a list
        for detection in detections:
            results.append([
                self.CLASSES[detection.class_id],
                detection.score,
                detection.xmin,
                detection.ymin,
                detection.xmax,
                detection.ymax,
            ])

        # 4. Draw detected boxes on the image
        annotated_image = self._draw_detections(image_bgr, detections)

        # 5. Encode the annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        postprocess_time = time.time() - start_post_time

        # Calculate total time
        total_time = preprocess_time + inference_time + postprocess_time

        return results, image_base64, {
            'preprocess_time': round(preprocess_time, 4),
            'inference_time': round(inference_time, 4),
            'postprocess_time': round(postprocess_time, 4),
            'total_time': round(total_time, 4),
        },
