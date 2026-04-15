"""
detector.py
-----------
MegaDetector Stage 1 — YOLOv8 bird localisation.

This script is called by the Node.js API server with a list of JPEG frame
paths as positional arguments.  It runs YOLOv8n on every frame and prints
a single JSON array to stdout that the Node.js parent process parses.

Output format (one JSON array, no other stdout):
[
  {
    "image": "/absolute/path/to/frame.jpg",
    "detections": [
      {
        "bbox": [x_norm, y_norm, w_norm, h_norm],  // 0-1, origin = top-left
        "confidence": 0.87,
        "label": "bird"
      }
    ]
  },
  ...
]

Usage (called by Node.js):
    python3 detector.py /tmp/frame_0000.jpg /tmp/frame_0001.jpg ...

Requirements (already installed in this project's venv):
    ultralytics, numpy
"""

import json
import sys
import os

# ── Constants ──────────────────────────────────────────────────────────────
BIRD_CLASS_ID      = 14          # COCO class index for "bird"
CONFIDENCE_THRESH  = 0.30        # Lower than main.py so we surface more proposals
                                  # for Gemini to classify in stage 2
MODEL_WEIGHTS      = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")


def load_yolo():
    """
    Load YOLOv8 model.  The ultralytics library auto-downloads weights on
    first run (~6 MB for yolov8n.pt) and caches them in ~/.cache/ultralytics.
    """
    try:
        from ultralytics import YOLO  # type: ignore
        # Suppress verbose output to keep stdout clean for JSON parsing.
        model = YOLO(MODEL_WEIGHTS, verbose=False)
        return model
    except Exception as exc:
        print(f"[detector.py] Failed to load YOLO: {exc}", file=sys.stderr)
        sys.exit(1)


def run_detection(model, frame_paths: list[str]) -> list[dict]:
    """
    Run YOLOv8 on all frames and return a list of per-frame result dicts.

    Only 'bird' class detections above CONFIDENCE_THRESH are included.
    Bounding boxes are normalised to [0, 1] in [x, y, w, h] format
    (top-left origin) to match the format expected by megadetector.ts.
    """
    results_out = []

    for path in frame_paths:
        frame_result = {"image": path, "detections": []}

        if not os.path.isfile(path):
            print(f"[detector.py] File not found, skipping: {path}", file=sys.stderr)
            results_out.append(frame_result)
            continue

        try:
            # Run inference on one frame; verbose=False keeps stdout clean.
            yolo_results = model(path, verbose=False, conf=CONFIDENCE_THRESH)
            result = yolo_results[0]

            # Image dimensions for normalisation.
            img_h, img_w = result.orig_shape[:2]

            for box in result.boxes:
                cls_id     = int(box.cls[0])
                confidence = float(box.conf[0])

                # Only keep bird detections.
                if cls_id != BIRD_CLASS_ID:
                    continue
                if confidence < CONFIDENCE_THRESH:
                    continue

                # xyxy → xywh normalised.
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                x_norm  = x1 / img_w
                y_norm  = y1 / img_h
                w_norm  = (x2 - x1) / img_w
                h_norm  = (y2 - y1) / img_h

                frame_result["detections"].append({
                    "bbox":       [
                        round(x_norm, 4),
                        round(y_norm, 4),
                        round(w_norm, 4),
                        round(h_norm, 4),
                    ],
                    "confidence": round(confidence, 4),
                    "label":      "bird",
                })

        except Exception as exc:
            print(f"[detector.py] Error on {path}: {exc}", file=sys.stderr)

        results_out.append(frame_result)

    return results_out


def main():
    frame_paths = sys.argv[1:]

    if not frame_paths:
        print(
            "[detector.py] No frame paths supplied. "
            "Usage: python3 detector.py frame1.jpg frame2.jpg ...",
            file=sys.stderr,
        )
        print("[]")   # empty JSON so caller doesn't crash
        sys.exit(0)

    print(
        f"[detector.py] Loading model '{MODEL_WEIGHTS}'...",
        file=sys.stderr,
    )
    model = load_yolo()
    print(
        f"[detector.py] Running detection on {len(frame_paths)} frame(s)...",
        file=sys.stderr,
    )

    results = run_detection(model, frame_paths)

    total_birds = sum(len(r["detections"]) for r in results)
    print(
        f"[detector.py] Done — {total_birds} bird proposals across "
        f"{len(results)} frames.",
        file=sys.stderr,
    )

    # Print ONLY the JSON to stdout — the Node.js parent reads this.
    print(json.dumps(results, separators=(",", ":")))


if __name__ == "__main__":
    main()
