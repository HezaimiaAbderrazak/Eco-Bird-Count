"""
video_analysis.py
-----------------
Full bird-detection pipeline for video files.

Pipeline
--------
  1. Extract every frame from the input video with OpenCV.
  2. At a configurable interval, run YOLOv8 to detect birds.
  3. The IoU tracker assigns a stable ID to each unique bird.
  4. Gemini 1.5 Flash classifies each unique track into a species
     (only once per bird, from its highest-confidence crop).
  5. Every output frame is annotated with:
       • Coloured bounding boxes (unique colour per track ID)
       • Species name + track ID + confidence
       • Live species tally panel (bottom-left)
       • Progress bar (top)
       • Unique bird count (top)
  6. The annotated video is written to disk.
  7. A species breakdown is printed to the console and optionally
     saved as a JSON results file.

Usage
-----
    python video_analysis.py --input birds.mp4
    python video_analysis.py --input birds.mp4 --interval 0.5
    python video_analysis.py --input birds.mp4 --model yolov8s.pt --output result.mp4
    python video_analysis.py --input birds.mp4 --no-species   # skip Gemini (faster)
    python video_analysis.py --input birds.mp4 --preview      # show live window
    python video_analysis.py --input birds.mp4 --json results.json

Press  Q / ESC  during --preview to stop early.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from detection import load_model, detect_animals
from tracker import IoUTracker
from species_id import classify_species, GEMINI_AVAILABLE


# ── Per-track colour palette (cycles for track IDs > 8) ───────────────────
_PALETTE = [
    (0,   200,  50),   # green
    (0,   160, 255),   # amber
    (200,   0, 200),   # magenta
    (0,   210, 210),   # cyan
    (255,  80,   0),   # orange
    (80,   80, 255),   # blue
    (0,   180, 120),   # teal
    (255, 200,   0),   # gold
]

WHITE = (255, 255, 255)
DARK  = (20,   20,  20)
GREEN = (0,   200,  50)


def _palette(track_id: int) -> tuple:
    return _PALETTE[track_id % len(_PALETTE)]


# ── Drawing helpers ────────────────────────────────────────────────────────

def _draw_box(frame: np.ndarray, track, species_cache: dict) -> None:
    """
    Draw a coloured bounding box with label directly on *frame* (in-place).
    Label format:  #ID  Species Name  conf%
    """
    x1, y1, x2, y2 = track.bbox
    colour = _palette(track.id)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    species   = species_cache.get(track.id, "Bird")
    label     = f"#{track.id}  {species}  {track.confidence:.0%}"
    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale, th = 0.55, 1
    (tw, fh), bl = cv2.getTextSize(label, font, scale, th)

    # Filled background pill above the box.
    cv2.rectangle(
        frame,
        (x1, y1 - fh - bl - 6),
        (x1 + tw + 8, y1),
        colour,
        cv2.FILLED,
    )
    cv2.putText(frame, label, (x1 + 4, y1 - bl - 3),
                font, scale, WHITE, th, cv2.LINE_AA)


def _draw_species_panel(frame: np.ndarray, species_counts: dict) -> None:
    """
    Draw a semi-transparent species-count panel in the bottom-left (in-place).
    """
    if not species_counts:
        return

    rows    = sorted(species_counts.items(), key=lambda x: -x[1])
    line_h  = 22
    pad     = 10
    panel_w = 300
    panel_h = 30 + len(rows) * line_h + pad
    h       = frame.shape[0]
    y_top   = h - panel_h - pad

    # Semi-transparent dark background.
    overlay = frame.copy()
    cv2.rectangle(overlay, (pad, y_top), (pad + panel_w, h - pad), DARK, cv2.FILLED)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    y = y_top + 22
    cv2.putText(frame, "Species Detected", (pad + 8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 230, 180), 1, cv2.LINE_AA)
    y += line_h

    for sp, cnt in rows:
        label = f"  {sp}: {cnt} bird{'s' if cnt != 1 else ''}"
        cv2.putText(frame, label, (pad + 8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, GREEN, 1, cv2.LINE_AA)
        y += line_h


def _draw_hud(
    frame: np.ndarray,
    frame_idx: int,
    total_frames: int,
    unique_birds: int,
    proc_fps: float,
) -> None:
    """
    Draw a progress bar and status line at the top of *frame* (in-place).
    """
    w = frame.shape[1]

    # Green progress bar (4 px tall).
    bar_w = int(w * frame_idx / max(total_frames, 1))
    cv2.rectangle(frame, (0, 0), (bar_w, 4), GREEN, cv2.FILLED)

    # Status text.
    pct  = frame_idx / max(total_frames, 1) * 100
    text = (
        f"Frame {frame_idx}/{total_frames}  ({pct:.0f}%)  |  "
        f"Unique birds: {unique_birds}  |  {proc_fps:.1f} fps"
    )
    cv2.putText(
        frame, text, (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.60, WHITE, 1, cv2.LINE_AA,
    )


# ── CLI ────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Eco-Bird-Count: full video bird detection, tracking & species ID"
    )
    p.add_argument(
        "--input", required=True, metavar="VIDEO",
        help="Path to input video file (MP4, AVI, MOV, MKV …)",
    )
    p.add_argument(
        "--output", default=None, metavar="VIDEO",
        help="Output annotated video path "
             "(default: <name>_annotated.mp4 next to input)",
    )
    p.add_argument(
        "--model", default="yolov8n.pt",
        help="YOLOv8 weights file (default: yolov8n.pt). "
             "Use yolov8s.pt / yolov8m.pt for better accuracy.",
    )
    p.add_argument(
        "--interval", type=float, default=1.0, metavar="SEC",
        help="Run detection every N seconds of video (default: 1.0). "
             "Lower = more detections but slower processing.",
    )
    p.add_argument(
        "--iou-threshold", type=float, default=0.25,
        help="IoU threshold for tracker matching (default: 0.25).",
    )
    p.add_argument(
        "--conf", type=float, default=0.40,
        help="YOLOv8 confidence threshold (default: 0.40).",
    )
    p.add_argument(
        "--no-species", action="store_true",
        help="Skip Gemini species identification — much faster.",
    )
    p.add_argument(
        "--preview", action="store_true",
        help="Show a live preview window while processing.",
    )
    p.add_argument(
        "--json", default=None, metavar="FILE",
        help="Save full results to a JSON file.",
    )
    return p.parse_args()


# ── Main pipeline ──────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> dict:
    # ── Validate input ─────────────────────────────────────────────────────
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[error] Input not found: {in_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        print(f"[error] Cannot open video: {in_path}")
        sys.exit(1)

    fps_in      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_fr    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s  = total_fr / fps_in

    print(f"\n{'═' * 60}")
    print(f"  EcoBird Video Analysis")
    print(f"{'─' * 60}")
    print(f"  Input  : {in_path.name}")
    print(f"  Size   : {frame_w}×{frame_h}  |  {fps_in:.1f} fps  |  "
          f"{duration_s:.1f}s  ({total_fr} frames)")
    print(f"  Model  : {args.model}")
    print(f"  Interval: every {args.interval}s")
    print(f"  Species ID: {'Gemini 1.5 Flash' if GEMINI_AVAILABLE and not args.no_species else 'disabled'}")
    print(f"{'═' * 60}\n")

    # ── Output video setup ─────────────────────────────────────────────────
    out_path = args.output or str(
        in_path.parent / (in_path.stem + "_annotated.mp4")
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps_in, (frame_w, frame_h))

    # ── Load model & tracker ───────────────────────────────────────────────
    model   = load_model(args.model)
    tracker = IoUTracker(
        iou_threshold=args.iou_threshold,
        max_age=int(fps_in * 2),   # track survives 2 s of occlusion
    )

    # ── State ──────────────────────────────────────────────────────────────
    species_cache: dict[int, str]  = {}          # track_id → species name
    species_counts: dict[str, int] = defaultdict(int)  # species → unique bird count
    last_visible: list             = []          # tracks visible in last analysed frame
    interval_frames = max(1, int(fps_in * args.interval))
    use_gemini      = GEMINI_AVAILABLE and not args.no_species

    frame_idx   = 0
    t_start     = time.perf_counter()

    print("  Processing… (press Q/ESC in preview window to stop early)\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            is_key_frame = (frame_idx == 1) or (frame_idx % interval_frames == 0)

            if is_key_frame:
                # ── Detect ────────────────────────────────────────────────
                detections    = detect_animals(model, frame)
                bird_dets     = [d for d in detections if d["is_bird"]]
                last_visible  = tracker.update(bird_dets, frame)

                # ── Species ID for newly created tracks ───────────────────
                if use_gemini:
                    for t in last_visible:
                        if t.id not in species_cache and t.best_crop_frame is not None:
                            print(
                                f"  [gemini] Track #{t.id} → classifying… ",
                                end="", flush=True,
                            )
                            sp = classify_species(t.best_crop_frame, t.best_crop_bbox or t.bbox)
                            species_cache[t.id]  = sp
                            t.species            = sp
                            t.classified         = True
                            print(sp)

                # Update species-level unique counts.
                for t in last_visible:
                    sp_name = species_cache.get(t.id, "Bird")
                    # We track unique IDs per species via a temporary set approach:
                    # species_counts holds total unique IDs found per species.
                    # Use the track ID itself — only add if not already counted.
                    pass   # handled below after the loop via all_tracks

            # ── Annotate frame ────────────────────────────────────────────
            for t in last_visible:
                _draw_box(frame, t, species_cache)

            # Rebuild species counts from all_tracks for the panel.
            _sp_display: dict[str, int] = defaultdict(int)
            _seen_ids: set[int] = set()
            for t in tracker.all_tracks():
                if t.id not in _seen_ids:
                    _seen_ids.add(t.id)
                    sp = species_cache.get(t.id, "Bird")
                    _sp_display[sp] += 1

            _draw_species_panel(frame, dict(_sp_display))

            elapsed  = time.perf_counter() - t_start
            proc_fps = frame_idx / max(elapsed, 1e-6)
            _draw_hud(frame, frame_idx, total_fr, tracker.total_unique, proc_fps)

            # ── Write to output video ──────────────────────────────────────
            writer.write(frame)

            # ── Optional live preview ──────────────────────────────────────
            if args.preview:
                cv2.imshow("EcoBird — Video Analysis  (Q to stop)", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    print("\n  [stopped by user]")
                    break

            # Progress to console (every 5 %).
            if frame_idx % max(1, total_fr // 20) == 0 or frame_idx == total_fr:
                pct = frame_idx / total_fr * 100
                eta = (elapsed / frame_idx) * (total_fr - frame_idx)
                print(
                    f"  {pct:5.1f}%  |  frame {frame_idx}/{total_fr}  "
                    f"|  unique birds: {tracker.total_unique}  |  ETA {eta:.0f}s",
                    flush=True,
                )

    finally:
        cap.release()
        writer.release()
        if args.preview:
            cv2.destroyAllWindows()

    # ── Final species counts ───────────────────────────────────────────────
    final_counts: dict[str, int] = defaultdict(int)
    for t in tracker.all_tracks():
        sp = species_cache.get(t.id, "Bird")
        final_counts[sp] += 1

    total_time = time.perf_counter() - t_start

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  Analysis Complete")
    print(f"{'─' * 60}")
    print(f"  Video duration      : {duration_s:.1f}s")
    print(f"  Processing time     : {total_time:.1f}s")
    print(f"  Unique birds found  : {tracker.total_unique}")
    print(f"{'─' * 60}")
    print(f"  {'Species':<38} {'Unique Birds':>10}")
    print(f"{'─' * 60}")
    for sp, cnt in sorted(final_counts.items(), key=lambda x: -x[1]):
        print(f"  {sp:<38} {cnt:>10}")
    print(f"{'═' * 60}")
    print(f"\n  Annotated video → {out_path}")

    # ── JSON output ────────────────────────────────────────────────────────
    results = {
        "input":              args.input,
        "output":             out_path,
        "duration_seconds":   duration_s,
        "processing_seconds": round(total_time, 2),
        "total_unique_birds": tracker.total_unique,
        "species_summary": dict(final_counts),
        "tracks": [
            {
                "id":      t.id,
                "species": species_cache.get(t.id, "Bird"),
                "hits":    t.hits,
            }
            for t in tracker.all_tracks()
        ],
    }

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        print(f"  JSON results    → {args.json}\n")
    else:
        print()

    return results


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = _parse_args()
    run(args)
