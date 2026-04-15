#!/usr/bin/env python3
"""
MegaDetector-style two-stage bird detection pipeline.

Architecture (mirrors MegaDetector's approach):
  Stage 1 — YOLO localization (primary):
      YOLOv8n.onnx detects WHERE birds are (fast bbox precision)
  Stage 1 — Saliency region proposals (fallback if ONNX model unavailable):
      Gradient-based region scoring finds high-activity zones for Gemini to examine
  Stage 2 — Classification (Node.js Gemini call):
      Gemini classifies the SPECIES inside each Stage-1 bounding box

Usage:
  python3 detector.py frame1.jpg frame2.jpg ...
Output:
  JSON array, one entry per input image:
  [{"image": "frame1.jpg", "detections": [{"bbox": [x, y, w, h], "confidence": 0.87, "label": "bird"}]}]
"""

import sys
import json
import os
import time
import numpy as np
from pathlib import Path
from PIL import Image as PILImage

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_PATH = Path(os.environ.get("YOLO_MODEL_PATH", "/tmp/yolov8n_ecobird.onnx"))
BIRD_CLASS_ID = 14          # COCO class 14 = "bird"
CONF_THRESHOLD = 0.18       # low threshold — catch partial/distant birds
IOU_THRESHOLD  = 0.45       # NMS overlap threshold
INPUT_SIZE     = 640        # YOLOv8 default

# Model download mirrors (tried in order)
MODEL_URLS = [
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx",
    "https://storage.googleapis.com/tfjs-models/savedmodel/ssd_mobilenet_v2/model.json",  # will 404 for onnx, skip
]

# ── Model download (try multiple sources) ─────────────────────────────────────

def try_download_model() -> bool:
    """Try to download YOLOv8n ONNX. Returns True on success."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return True
    try:
        import requests
        mirrors = [
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx",
            "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx",
        ]
        for url in mirrors:
            try:
                print(f"[Stage1] Trying {url}", file=sys.stderr, flush=True)
                r = requests.get(url, timeout=30, stream=True)
                if r.status_code == 200 and int(r.headers.get("content-length", 0)) > 1_000_000:
                    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                    with open(MODEL_PATH, "wb") as f:
                        for chunk in r.iter_content(chunk_size=65536):
                            f.write(chunk)
                    print(f"[Stage1] Model downloaded ({MODEL_PATH.stat().st_size // 1024} KB)", file=sys.stderr, flush=True)
                    return True
            except Exception as e:
                print(f"[Stage1] Mirror failed: {e}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[Stage1] Download skipped: {e}", file=sys.stderr, flush=True)
    return False


# ── YOLO inference path ───────────────────────────────────────────────────────

_session = None

def get_session():
    global _session
    if _session is not None:
        return _session
    if not try_download_model():
        return None
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 4
        _session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        print("[Stage1] YOLO ONNX session ready", file=sys.stderr, flush=True)
        return _session
    except Exception as e:
        print(f"[Stage1] ONNX session failed: {e}", file=sys.stderr, flush=True)
        return None


def letterbox(img_array: np.ndarray, target: int = INPUT_SIZE):
    h, w = img_array.shape[:2]
    scale = target / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    pil = PILImage.fromarray(img_array).resize((new_w, new_h), PILImage.BILINEAR)
    resized = np.array(pil)
    pad_top  = (target - new_h) // 2
    pad_left = (target - new_w) // 2
    canvas = np.full((target, target, 3), 114, dtype=np.uint8)
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas, scale, (pad_top, pad_left)


def decode_yolov8(raw: np.ndarray) -> tuple:
    pred = raw[0].T                   # [8400, 84]
    boxes_xywh  = pred[:, :4]
    bird_scores = pred[:, 4 + BIRD_CLASS_ID]
    mask = bird_scores > CONF_THRESHOLD
    if not mask.any():
        return np.empty((0, 4)), np.empty(0)
    boxes_xywh = boxes_xywh[mask]
    scores      = bird_scores[mask]
    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    return np.stack([x1, y1, x2, y2], axis=1), scores


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = IOU_THRESHOLD) -> list:
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        if order.size == 1: break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area_i + area_j - inter + 1e-6)
        order = order[1:][iou < iou_thr]
    return keep


def yolo_detect(img: np.ndarray, orig_w: int, orig_h: int) -> list[dict]:
    sess = get_session()
    if sess is None:
        return []
    padded, scale, (pad_top, pad_left) = letterbox(img)
    blob = padded.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
    raw = sess.run(None, {sess.get_inputs()[0].name: blob})
    boxes_xyxy, scores = decode_yolov8(raw[0])
    if len(boxes_xyxy) == 0:
        return []
    keep = nms(boxes_xyxy, scores)
    boxes_xyxy = boxes_xyxy[keep]; scores = scores[keep]
    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_left) / scale
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_top)  / scale
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)
    results = []
    for box, score in zip(boxes_xyxy, scores):
        x1, y1, x2, y2 = box
        bw, bh = (x2 - x1) / orig_w, (y2 - y1) / orig_h
        if bw < 0.005 or bh < 0.005:
            continue
        results.append({
            "bbox":       [round(float(x1/orig_w),4), round(float(y1/orig_h),4), round(float(bw),4), round(float(bh),4)],
            "confidence": round(float(score), 4),
            "label":      "bird",
        })
    return results


# ── Saliency-based region proposal fallback ───────────────────────────────────
# When YOLO is unavailable, we use gradient-based spatial saliency scoring to
# find high-activity image regions — the same concept as Selective Search or
# EdgeBoxes, implemented with only numpy + Pillow.
#
# Why this works for birds:
#   Birds have sharp edges vs. sky/water backgrounds, distinct colors, and
#   compact shapes. High spatial-gradient regions in an image frame correspond
#   to objects of interest. We tile the image into a 4×4 grid, score each tile
#   by gradient magnitude (using a Sobel approximation), then return the top
#   tiles as "potential bird" bounding boxes for Gemini to classify.

GRID_ROWS = 4
GRID_COLS = 4
TOP_K_TILES = 6        # max tiles to report
MIN_TILE_SCORE = 0.30  # normalised gradient threshold (0-1)


def sobel_score(gray: np.ndarray) -> float:
    """Mean gradient magnitude of a grayscale patch, normalised to [0,1]."""
    if gray.size == 0:
        return 0.0
    # Sobel-X approximation via numpy diff
    gx = np.abs(np.diff(gray.astype(np.float32), axis=1)).mean()
    gy = np.abs(np.diff(gray.astype(np.float32), axis=0)).mean()
    score = (gx + gy) / 2.0
    return float(np.clip(score / 80.0, 0, 1))  # 80 is a typical mid-grey gradient


def saliency_proposals(img: np.ndarray, orig_w: int, orig_h: int) -> list[dict]:
    """
    Divide frame into GRID_ROWS × GRID_COLS tiles.
    Score each tile by Sobel gradient magnitude and colour saturation.
    Return top-K tiles as normalised [x, y, w, h] bounding boxes.
    """
    # Downsample for speed (max 320px on long side)
    scale = min(1.0, 320 / max(orig_w, orig_h))
    small_w, small_h = int(orig_w * scale), int(orig_h * scale)
    small = np.array(PILImage.fromarray(img).resize((small_w, small_h), PILImage.BILINEAR))

    # Grayscale
    gray = (0.2126 * small[:, :, 0] + 0.7152 * small[:, :, 1] + 0.0722 * small[:, :, 2])

    # HSV saturation approximation: max(R,G,B) - min(R,G,B)
    sat = small.max(axis=2).astype(np.float32) - small.min(axis=2).astype(np.float32)

    tile_h = small_h / GRID_ROWS
    tile_w = small_w / GRID_COLS
    tiles = []

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            r0, r1 = int(row * tile_h), int((row + 1) * tile_h)
            c0, c1 = int(col * tile_w), int((col + 1) * tile_w)
            patch_gray = gray[r0:r1, c0:c1]
            patch_sat  = sat[r0:r1, c0:c1]

            grad_s = sobel_score(patch_gray)
            sat_s  = float(np.clip(patch_sat.mean() / 128.0, 0, 1))

            combined = 0.65 * grad_s + 0.35 * sat_s
            if combined >= MIN_TILE_SCORE:
                # Convert to original-space normalised bbox
                bx = (col * tile_w) / small_w
                by = (row * tile_h) / small_h
                bw = tile_w / small_w
                bh = tile_h / small_h
                tiles.append((combined, bx, by, bw, bh))

    # Sort by score, keep top-K
    tiles.sort(key=lambda t: -t[0])
    results = []
    for score, bx, by, bw, bh in tiles[:TOP_K_TILES]:
        results.append({
            "bbox":       [round(bx, 4), round(by, 4), round(bw, 4), round(bh, 4)],
            "confidence": round(score, 4),
            "label":      "region",  # not confirmed bird — Gemini will verify
        })
    return results


# ── Main detect function ───────────────────────────────────────────────────────

def detect_image(image_path: str) -> list[dict]:
    try:
        img = np.array(PILImage.open(image_path).convert("RGB"))
    except Exception as e:
        print(f"[Stage1] Cannot open {image_path}: {e}", file=sys.stderr)
        return []

    orig_h, orig_w = img.shape[:2]

    # Primary: YOLO-based detection
    detections = yolo_detect(img, orig_w, orig_h)

    # Fallback: saliency region proposals
    if not detections:
        detections = saliency_proposals(img, orig_w, orig_h)
        if detections:
            print(f"[Stage1] YOLO unavailable — using {len(detections)} saliency proposals for {image_path}", file=sys.stderr, flush=True)

    return detections


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    image_paths = sys.argv[1:]
    if not image_paths:
        print("[]")
        sys.exit(0)

    output = []
    for path in image_paths:
        detections = detect_image(path)
        output.append({"image": path, "detections": detections})

    print(json.dumps(output))
