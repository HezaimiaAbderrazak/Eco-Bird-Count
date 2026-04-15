"""
species_id.py
-------------
Gemini 1.5 Flash-powered bird species identification.

For each unique tracked bird, one high-quality crop is sent to Gemini
with an expert ornithologist prompt.  Results are returned as a plain
English common name (e.g. "Barn Swallow", "Eurasian Hoopoe").

If the GEMINI_API_KEY environment variable is not set, or if the
`google-generativeai` package is not installed, the module degrades
gracefully to returning "Bird" for every detection.
"""

import base64
import os
import time

import cv2
import numpy as np

# ── Conditional Gemini import ──────────────────────────────────────────────
GEMINI_AVAILABLE = False
_gemini_model = None

try:
    import google.generativeai as genai

    _api_key = os.environ.get("GEMINI_API_KEY", "")
    if _api_key:
        genai.configure(api_key=_api_key)
        _gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        GEMINI_AVAILABLE = True
        print("[species_id] Gemini 1.5 Flash ready for species identification.")
    else:
        print(
            "[species_id] GEMINI_API_KEY not set — species ID disabled. "
            "Set the env var to enable it."
        )
except ImportError:
    print(
        "[species_id] google-generativeai not installed. "
        "Run: pip install google-generativeai"
    )


# ── Prompt (expert ornithologist context) ──────────────────────────────────
_SPECIES_PROMPT = """You are an expert ornithologist with deep knowledge of North African, 
Mediterranean, and European bird species.

Study this bird image carefully and identify the species.

Reply with ONLY the common English name of the species 
(e.g. 'Barn Swallow', 'Eurasian Hoopoe', 'Common Starling').

Rules:
- If you can see the bird clearly → give the specific species name.
- If the bird is partly visible but identifiable → give your best guess.
- If the image is too blurry or no bird is visible → reply exactly: Unknown bird
- Do NOT include explanations, qualifications, or extra text.
- Do NOT use markdown formatting."""


# ── Internal helpers ───────────────────────────────────────────────────────

def _crop_and_encode(frame: np.ndarray, bbox: tuple, pad: int = 24) -> str | None:
    """
    Crop the bird bounding-box region (with padding) and encode it as a
    base-64 JPEG string suitable for Gemini's inline_data format.

    Returns None if the crop is too small to be useful (<16×16 px).
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
        return None

    # Upscale tiny crops so Gemini can see more detail.
    crop_h, crop_w = crop.shape[:2]
    if crop_h < 128 or crop_w < 128:
        scale = max(128 / crop_h, 128 / crop_w)
        new_w = int(crop_w * scale)
        new_h = int(crop_h * scale)
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ── Public API ─────────────────────────────────────────────────────────────

def classify_species(frame: np.ndarray, bbox: tuple, retries: int = 2) -> str:
    """
    Identify the bird species in the given frame crop using Gemini.

    Args:
        frame:   Full BGR video frame (numpy array).
        bbox:    Bird bounding box (x1, y1, x2, y2).
        retries: Number of retry attempts on API failure.

    Returns:
        Common English species name, or "Bird" if classification fails.
    """
    if not GEMINI_AVAILABLE or _gemini_model is None:
        return "Bird"

    img_b64 = _crop_and_encode(frame, bbox)
    if img_b64 is None:
        return "Bird"

    for attempt in range(1, retries + 2):
        try:
            response = _gemini_model.generate_content(
                [
                    _SPECIES_PROMPT,
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": img_b64,
                        }
                    },
                ]
            )
            name = response.text.strip().strip("'\"").strip()
            # Sanity check: reject suspiciously long or empty responses.
            if name and len(name) < 80:
                return name
            return "Bird"

        except Exception as exc:
            if attempt <= retries:
                wait = 2 ** attempt
                print(f"\n[species_id] Gemini error (attempt {attempt}): {exc}. "
                      f"Retrying in {wait}s…")
                time.sleep(wait)
            else:
                print(f"\n[species_id] Gemini failed after {retries+1} attempts: {exc}")
                return "Bird"

    return "Bird"
