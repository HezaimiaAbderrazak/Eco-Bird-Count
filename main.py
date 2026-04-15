"""
main.py
-------
Entry point for the Eco-Bird-Count real-time detection system.

Runs a YOLOv8 model on your webcam feed, highlights birds and other
animals with bounding boxes, shows a per-frame bird count, and
(optionally) enriches detections with eBird occurrence data.

Usage
-----
    python main.py                    # webcam, no eBird
    python main.py --ebird-key TOKEN  # webcam + eBird nearby lookup
    python main.py --source video.mp4 # run on a saved video file

Press  Q  or  ESC  to quit.
"""

import argparse
import time
import sys

import cv2

from detection import load_model, detect_animals
from utils import (
    count_birds,
    count_by_label,
    draw_detections,
    draw_hud,
    get_ebird_nearby_observations,
    format_ebird_observation,
    match_detection_to_ebird,
)


# ── Configuration defaults ─────────────────────────────────────────────────
DEFAULT_MODEL   = "yolov8n.pt"   # nano – fastest; swap for yolov8s.pt for more accuracy
DEFAULT_SOURCE  = 0              # 0 = first webcam; pass a filepath for a video file

# eBird: set your coordinates here or pass them via CLI flags.
DEFAULT_LAT = 36.7372            # Default: Algiers, Algeria
DEFAULT_LON = 3.0869


# ── Argument parsing ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eco-Bird-Count: real-time bird detection with YOLOv8 + eBird"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="YOLOv8 weights file (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--source", default=DEFAULT_SOURCE,
        help="Video source: 0/1/2 for webcam index, or a video file path",
    )
    parser.add_argument(
        "--ebird-key", default=None, metavar="TOKEN",
        help="eBird API key. If provided, nearby species data is fetched on startup.",
    )
    parser.add_argument(
        "--lat", type=float, default=DEFAULT_LAT,
        help="Latitude for eBird nearby lookup (default: Algiers)",
    )
    parser.add_argument(
        "--lon", type=float, default=DEFAULT_LON,
        help="Longitude for eBird nearby lookup (default: Algiers)",
    )
    parser.add_argument(
        "--ebird-dist", type=int, default=25,
        help="eBird search radius in km (default: 25)",
    )
    return parser.parse_args()


# ── eBird pre-fetch ────────────────────────────────────────────────────────

def fetch_ebird_context(args: argparse.Namespace) -> list[dict]:
    """
    If an eBird API key was supplied, fetch nearby observations once at
    startup and return them.  They are used to annotate detections in the
    console (not drawn on-screen to keep the UI clean).
    """
    if not args.ebird_key:
        return []

    print(
        f"\n[eBird] Fetching recent observations near "
        f"({args.lat:.4f}, {args.lon:.4f}) within {args.ebird_dist} km …"
    )
    observations = get_ebird_nearby_observations(
        api_key=args.ebird_key,
        lat=args.lat,
        lon=args.lon,
        dist_km=args.ebird_dist,
    )

    if observations:
        print(f"[eBird] {len(observations)} recent sightings found:")
        for obs in observations[:5]:          # print first five as a preview
            print(f"        • {format_ebird_observation(obs)}")
    else:
        print("[eBird] No observations returned (check your API key and location).")

    return observations


# ── Main loop ──────────────────────────────────────────────────────────────

def run(args: argparse.Namespace, ebird_obs: list[dict]) -> None:
    """
    Open the video source and run the detection loop until the user quits.
    """
    # ── Open video source ──────────────────────────────────────────────────
    # Allow passing an integer index or a string file path.
    source = args.source
    try:
        source = int(source)          # webcam index
    except (ValueError, TypeError):
        pass                          # keep as string file path

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(
            f"\n[main] ERROR: Could not open video source '{source}'.\n"
            "       • For a webcam, make sure it is connected and not in use.\n"
            "       • For a file, check that the path is correct."
        )
        sys.exit(1)

    print(f"\n[main] Video source opened: {source}")
    print("[main] Press  Q  or  ESC  to quit.\n")

    # ── Load model ─────────────────────────────────────────────────────────
    model = load_model(args.model)

    # ── FPS tracking ──────────────────────────────────────────────────────
    prev_time    = time.perf_counter()
    fps_smoothed = 0.0
    ALPHA        = 0.1               # exponential moving average weight

    # ── Session-level totals ───────────────────────────────────────────────
    session_bird_count = 0           # running total of birds seen across frames
    frame_index        = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # End of video file, or webcam disconnected.
                print("[main] Video stream ended.")
                break

            frame_index += 1

            # ── Detect ────────────────────────────────────────────────────
            detections = detect_animals(model, frame)
            bird_count = count_birds(detections)
            tally      = count_by_label(detections)

            session_bird_count += bird_count

            # ── eBird contextual print (every 30 frames to avoid spam) ────
            if ebird_obs and bird_count > 0 and frame_index % 30 == 0:
                for d in detections:
                    if d["is_bird"]:
                        match = match_detection_to_ebird(d["label"], ebird_obs)
                        if match:
                            print(
                                f"[eBird match] {format_ebird_observation(match)}"
                            )

            # ── Draw ──────────────────────────────────────────────────────
            annotated = draw_detections(frame, detections)

            # FPS: exponential moving average for a smooth readout.
            now          = time.perf_counter()
            instant_fps  = 1.0 / max(now - prev_time, 1e-6)
            fps_smoothed = ALPHA * instant_fps + (1 - ALPHA) * fps_smoothed
            prev_time    = now

            annotated = draw_hud(annotated, bird_count, tally, fps_smoothed)

            # ── Show ──────────────────────────────────────────────────────
            cv2.imshow("Eco-Bird-Count  |  Q to quit", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):   # Q or ESC
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # ── Session summary ────────────────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print(f"  Session complete  |  {frame_index} frames processed")
    print(f"  Total birds detected across all frames: {session_bird_count}")
    print(f"{'─' * 50}\n")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args       = parse_args()
    ebird_obs  = fetch_ebird_context(args)
    run(args, ebird_obs)
