"""
detection.py
------------
Handles all YOLOv8-based object detection logic.
Loads the model once and exposes a function that runs inference
on a single frame, returning structured detection results.
"""

from ultralytics import YOLO
import numpy as np

# COCO class index for "bird" (class 14).
# YOLOv8 trained on COCO can detect many animals; we highlight birds specially.
BIRD_CLASS_ID = 14

# All COCO animal class IDs we care about displaying (bird + common wildlife).
ANIMAL_CLASS_IDS = {
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
}

# Minimum confidence score to accept a detection (0–1).
CONFIDENCE_THRESHOLD = 0.40


def load_model(weights: str = "yolov8n.pt") -> YOLO:
    """
    Load a YOLOv8 model from a weights file.

    The first time this runs it automatically downloads the weights
    from the Ultralytics CDN (~6 MB for 'yolov8n.pt').

    Args:
        weights: Model variant filename.
                 Options: yolov8n.pt (fastest) → yolov8s/m/l/x.pt (more accurate)

    Returns:
        A loaded YOLO model instance ready for inference.
    """
    print(f"[detection] Loading YOLOv8 model: {weights}")
    model = YOLO(weights)
    print("[detection] Model loaded successfully.")
    return model


def detect_animals(model: YOLO, frame: np.ndarray) -> list[dict]:
    """
    Run YOLOv8 inference on a single BGR frame (as returned by OpenCV).

    Only detections whose class is in ANIMAL_CLASS_IDS and whose confidence
    exceeds CONFIDENCE_THRESHOLD are returned.

    Args:
        model:  Loaded YOLO model instance.
        frame:  A numpy array of shape (H, W, 3) in BGR colour space.

    Returns:
        A list of dicts, each containing:
            - label     (str):  Human-readable class name, e.g. "bird"
            - class_id  (int):  COCO class index
            - conf      (float): Confidence score 0–1
            - bbox      (tuple): (x1, y1, x2, y2) in pixel coordinates
            - is_bird   (bool): True if the detected class is a bird
    """
    # Run inference; verbose=False keeps the console clean each frame.
    results = model(frame, verbose=False)[0]

    detections = []
    for box in results.boxes:
        class_id = int(box.cls[0])

        # Skip classes we are not interested in.
        if class_id not in ANIMAL_CLASS_IDS:
            continue

        confidence = float(box.conf[0])

        # Skip low-confidence detections.
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detections.append({
            "label":    ANIMAL_CLASS_IDS[class_id],
            "class_id": class_id,
            "conf":     confidence,
            "bbox":     (x1, y1, x2, y2),
            "is_bird":  class_id == BIRD_CLASS_ID,
        })

    return detections
