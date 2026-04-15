"""
tracker.py
----------
IoU-based multi-object tracker for bird detection.

Assigns a stable, globally-unique ID to each bird across frames so
that the same individual is not counted twice even if it leaves and
re-enters the frame (within a short gap).

Algorithm
---------
For each new set of detections:
  1. Compute pairwise IoU between every active track and every detection.
  2. Greedily match detections to tracks (highest IoU first).
  3. Unmatched detections → new tracks.
  4. Tracks not matched for `max_age` consecutive frames → removed.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── IoU helper ─────────────────────────────────────────────────────────────

def compute_iou(boxA: tuple, boxB: tuple) -> float:
    """
    Compute Intersection-over-Union between two (x1, y1, x2, y2) boxes.

    Returns a float in [0, 1].  Returns 0.0 for degenerate boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter   = inter_w * inter_h

    areaA = max(0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    union = areaA + areaB - inter

    return inter / union if union > 0 else 0.0


# ── Track dataclass ────────────────────────────────────────────────────────

@dataclass
class Track:
    """Represents a single tracked bird across one or more frames."""
    id:           int
    bbox:         tuple           # (x1, y1, x2, y2) — most recent position
    label:        str             # YOLO class label (usually "bird")
    confidence:   float           # Most recent detection confidence
    age:          int   = 0      # Frames since last matched (0 = visible now)
    hits:         int   = 1      # Total frames where this track was matched
    classified:   bool  = False  # True once species ID has been obtained
    species:      Optional[str] = None
    # Best frame + bbox for Gemini classification (highest-confidence crop):
    best_crop_frame: Optional[object] = None   # numpy ndarray
    best_crop_bbox:  Optional[tuple]  = None


# ── Tracker ────────────────────────────────────────────────────────────────

class IoUTracker:
    """
    Simple greedy IoU multi-object tracker.

    Args:
        iou_threshold: Minimum IoU to match a detection to an existing track.
                       Lower → looser matching (more merges).
                       Higher → stricter (more new IDs created).
        max_age:       Number of consecutive unmatched frames before a track
                       is deleted.  Higher → tracks survive longer occlusions.
    """

    def __init__(self, iou_threshold: float = 0.25, max_age: int = 8):
        self.iou_threshold = iou_threshold
        self.max_age       = max_age
        self._tracks: list[Track] = []
        self._all_tracks: list[Track] = []   # keeps deleted tracks for summary
        self._next_id: int = 1

    # ── Public API ─────────────────────────────────────────────────────────

    def update(
        self,
        detections: list[dict],
        frame,                          # BGR numpy array
    ) -> list[Track]:
        """
        Ingest new detections, update tracks, and return the set of tracks
        that are currently visible (age == 0).

        Args:
            detections: List of detection dicts (from detection.detect_animals).
                        Only bird detections should be passed in.
            frame:      The current BGR frame — stored as a crop candidate for
                        species classification.

        Returns:
            List of Track objects visible in *this* frame.
        """
        # Age all existing tracks by 1 (reset to 0 when matched).
        for t in self._tracks:
            t.age += 1

        unmatched = list(detections)

        # ── Greedy matching ────────────────────────────────────────────────
        for track in self._tracks:
            if not unmatched:
                break

            ious = [compute_iou(track.bbox, d["bbox"]) for d in unmatched]
            best_idx = int(np.argmax(ious))
            best_iou = ious[best_idx]

            if best_iou >= self.iou_threshold:
                det = unmatched.pop(best_idx)
                track.bbox       = det["bbox"]
                track.confidence = det["conf"]
                track.age        = 0
                track.hits      += 1

                # Keep the highest-confidence crop for Gemini.
                if not track.classified:
                    if (track.best_crop_frame is None or
                            det["conf"] > track.confidence):
                        track.best_crop_frame = frame
                        track.best_crop_bbox  = det["bbox"]

        # ── New tracks for unmatched detections ───────────────────────────
        for det in unmatched:
            t = Track(
                id=self._next_id,
                bbox=det["bbox"],
                label=det["label"],
                confidence=det["conf"],
                best_crop_frame=frame,
                best_crop_bbox=det["bbox"],
            )
            self._next_id += 1
            self._tracks.append(t)
            self._all_tracks.append(t)

        # ── Remove expired tracks ─────────────────────────────────────────
        self._tracks = [t for t in self._tracks if t.age <= self.max_age]

        # Return only the tracks visible in this frame.
        return [t for t in self._tracks if t.age == 0]

    def all_tracks(self) -> list[Track]:
        """All tracks ever created, including expired ones."""
        return self._all_tracks

    @property
    def total_unique(self) -> int:
        """Total number of unique track IDs assigned so far."""
        return self._next_id - 1
