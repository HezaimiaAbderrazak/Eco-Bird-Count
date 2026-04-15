"""
utils.py
--------
Helper functions for drawing detections on frames, counting birds,
and optionally querying the eBird API for species occurrence data.
"""

import cv2
import requests
import numpy as np
from datetime import datetime

# ── Drawing colours (BGR) ──────────────────────────────────────────────────
COLOUR_BIRD  = (0, 200, 50)    # vivid green  – birds
COLOUR_OTHER = (50, 170, 255)  # amber        – other animals
COLOUR_TEXT  = (255, 255, 255) # white text

# ── eBird API ──────────────────────────────────────────────────────────────
EBIRD_BASE_URL = "https://api.ebird.org/v2"


# ── Counting ───────────────────────────────────────────────────────────────

def count_birds(detections: list[dict]) -> int:
    """
    Return the number of detections whose label is 'bird'.

    Args:
        detections: List of detection dicts produced by detection.detect_animals().

    Returns:
        Integer count of birds in the current frame.
    """
    return sum(1 for d in detections if d["is_bird"])


def count_by_label(detections: list[dict]) -> dict[str, int]:
    """
    Return a tally of every detected class in the current frame.

    Example return value: {"bird": 3, "cat": 1}
    """
    tally: dict[str, int] = {}
    for d in detections:
        tally[d["label"]] = tally.get(d["label"], 0) + 1
    return tally


# ── Drawing ────────────────────────────────────────────────────────────────

def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """
    Draw bounding boxes and confidence labels on a copy of *frame*.

    Birds are drawn in green; other animals in amber.  The box thickness
    is slightly heavier for birds so they stand out at a glance.

    Args:
        frame:      BGR numpy array (unmodified).
        detections: Output of detection.detect_animals().

    Returns:
        A new numpy array with annotations drawn on it.
    """
    annotated = frame.copy()

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        colour    = COLOUR_BIRD if d["is_bird"] else COLOUR_OTHER
        thickness = 3 if d["is_bird"] else 2

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, thickness)

        # Label text: "bird 87%"
        label_text = f"{d['label']}  {d['conf']:.0%}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )

        # Filled background pill so the text is readable on any background.
        cv2.rectangle(
            annotated,
            (x1, y1 - text_h - baseline - 6),
            (x1 + text_w + 6, y1),
            colour,
            cv2.FILLED,
        )
        cv2.putText(
            annotated,
            label_text,
            (x1 + 3, y1 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOUR_TEXT,
            1,
            cv2.LINE_AA,
        )

    return annotated


def draw_hud(
    frame: np.ndarray,
    bird_count: int,
    tally: dict[str, int],
    fps: float,
) -> np.ndarray:
    """
    Overlay a heads-up display in the top-left corner of the frame.

    Shows:
      • Current FPS
      • Bird count for this frame
      • Full per-class tally

    Args:
        frame:      Annotated BGR frame (output of draw_detections).
        bird_count: Number of birds detected this frame.
        tally:      Per-class counts dict.
        fps:        Current frames-per-second estimate.

    Returns:
        Frame with HUD drawn on it (in-place modification).
    """
    # Semi-transparent dark panel
    overlay = frame.copy()
    panel_h = 30 + len(tally) * 22 + 30
    cv2.rectangle(overlay, (10, 10), (260, 10 + panel_h), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = 38
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (18, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA,
    )
    y += 26
    cv2.putText(
        frame, f"Birds this frame: {bird_count}", (18, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOUR_BIRD, 2, cv2.LINE_AA,
    )
    y += 28

    for label, cnt in sorted(tally.items()):
        colour = COLOUR_BIRD if label == "bird" else COLOUR_OTHER
        cv2.putText(
            frame, f"  {label}: {cnt}", (18, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA,
        )
        y += 22

    # Timestamp bottom-right
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    (tw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    h, w = frame.shape[:2]
    cv2.putText(
        frame, ts, (w - tw - 10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA,
    )

    return frame


# ── eBird API helpers ──────────────────────────────────────────────────────

def get_ebird_nearby_observations(
    api_key: str,
    lat: float,
    lon: float,
    dist_km: int = 25,
    max_results: int = 10,
) -> list[dict]:
    """
    Fetch recent bird observations near a GPS coordinate from the eBird API.

    Requires a free eBird API key: https://ebird.org/api/keygen

    Args:
        api_key:     Your personal eBird API token.
        lat:         Latitude  (decimal degrees).
        lon:         Longitude (decimal degrees).
        dist_km:     Search radius in kilometres (1–50).
        max_results: Maximum number of results to return.

    Returns:
        List of observation dicts with keys:
            comName, sciName, locName, obsDt, howMany
        Returns an empty list on any error.
    """
    url = f"{EBIRD_BASE_URL}/data/obs/geo/recent"
    headers = {"X-eBirdApiToken": api_key}
    params = {
        "lat":      lat,
        "lng":      lon,
        "dist":     dist_km,
        "maxResults": max_results,
        "fmt":      "json",
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=8)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        print(f"[utils] eBird API request failed: {exc}")
        return []


def match_detection_to_ebird(
    common_name: str,
    ebird_observations: list[dict],
) -> dict | None:
    """
    Try to find an eBird observation that matches a detected bird's common name.

    Does a simple case-insensitive substring match between the YOLO label
    (e.g. "bird") and eBird common names (e.g. "European Robin").
    YOLO's generic "bird" label will match any eBird entry; more specific
    labels (from a fine-tuned model) will match more precisely.

    Args:
        common_name:        Detected label string from YOLO.
        ebird_observations: List returned by get_ebird_nearby_observations().

    Returns:
        The first matching eBird observation dict, or None.
    """
    name_lower = common_name.lower()
    for obs in ebird_observations:
        if name_lower in obs.get("comName", "").lower() or \
           name_lower == "bird":          # generic match for any species
            return obs
    return None


def format_ebird_observation(obs: dict) -> str:
    """
    Format a single eBird observation into a readable one-line string.

    Example output:
        "European Robin (Erithacus rubecula) – 3 seen at Regent's Park on 2024-04-12"
    """
    name   = obs.get("comName", "Unknown")
    sci    = obs.get("sciName", "")
    loc    = obs.get("locName", "unknown location")
    date   = obs.get("obsDt",  "unknown date")[:10]  # keep YYYY-MM-DD only
    count  = obs.get("howMany", "?")
    return f"{name} ({sci}) – {count} seen at {loc} on {date}"
