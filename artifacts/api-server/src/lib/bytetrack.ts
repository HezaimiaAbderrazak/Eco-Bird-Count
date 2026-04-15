/**
 * ByteTrack-inspired multi-object tracker implemented in TypeScript.
 *
 * Core algorithm:
 *  1. For each frame, compute IoU between all detections and active tracks.
 *  2. Greedy assignment: highest-IoU pairs matched first (O(n²) — sufficient for bird counts).
 *  3. Unmatched high-confidence detections → new tracks.
 *  4. Unmatched tracks → "lost" state; kept for MAX_AGE frames via predicted position.
 *  5. Recovered lost tracks retain their original track ID — this is the key to ID continuity.
 *
 * Design note: We deliberately skip the Kalman filter's covariance update for simplicity;
 * constant-velocity linear extrapolation is sufficient for slow-moving bird footage.
 */

export interface RawDetection {
  species: string;
  bbox: [number, number, number, number]; // [x, y, w, h] normalised 0–1
  confidence: number;
  count?: number; // estimated flock/group size (1 for individuals)
}

export interface TrackedDetection extends RawDetection {
  trackId: number;
  isConfirmed: boolean;
  framesLost: number;
}

interface Track {
  trackId: number;
  species: string;
  bbox: [number, number, number, number];
  velocity: [number, number]; // [vx, vy] per frame
  confidence: number;
  age: number;       // total frames seen
  hits: number;      // consecutive matched frames
  framesLost: number;
  state: "active" | "lost";
}

const IOU_THRESHOLD_HIGH = 0.25;  // match active tracks
const IOU_THRESHOLD_LOW  = 0.15;  // match lost tracks (re-identification)
const MAX_AGE            = 5;     // frames a track survives without a match
const MIN_HITS_TO_CONFIRM = 1;    // frames needed to confirm a new track

let _nextTrackId = 1;

function computeIoU(
  a: [number, number, number, number],
  b: [number, number, number, number],
): number {
  const ax1 = a[0], ay1 = a[1], ax2 = a[0] + a[2], ay2 = a[1] + a[3];
  const bx1 = b[0], by1 = b[1], bx2 = b[0] + b[2], by2 = b[1] + b[3];

  const ix1 = Math.max(ax1, bx1);
  const iy1 = Math.max(ay1, by1);
  const ix2 = Math.min(ax2, bx2);
  const iy2 = Math.min(ay2, by2);

  if (ix2 <= ix1 || iy2 <= iy1) return 0;

  const inter = (ix2 - ix1) * (iy2 - iy1);
  const aArea = a[2] * a[3];
  const bArea = b[2] * b[3];
  return inter / (aArea + bArea - inter + 1e-7);
}

function predictBbox(track: Track): [number, number, number, number] {
  return [
    Math.max(0, Math.min(1 - track.bbox[2], track.bbox[0] + track.velocity[0])),
    Math.max(0, Math.min(1 - track.bbox[3], track.bbox[1] + track.velocity[1])),
    track.bbox[2],
    track.bbox[3],
  ];
}

function greedyMatch(
  detBboxes: [number, number, number, number][],
  trkBboxes: [number, number, number, number][],
  threshold: number,
): { matches: [number, number][]; unmatchedDets: number[]; unmatchedTrks: number[] } {
  const matched: [number, number][] = [];
  const matchedDets = new Set<number>();
  const matchedTrks = new Set<number>();

  // Build full IoU matrix and sort pairs descending
  const pairs: { iou: number; di: number; ti: number }[] = [];
  for (let di = 0; di < detBboxes.length; di++) {
    for (let ti = 0; ti < trkBboxes.length; ti++) {
      const score = computeIoU(detBboxes[di]!, trkBboxes[ti]!);
      if (score >= threshold) pairs.push({ iou: score, di, ti });
    }
  }
  pairs.sort((a, b) => b.iou - a.iou);

  for (const { di, ti } of pairs) {
    if (!matchedDets.has(di) && !matchedTrks.has(ti)) {
      matched.push([di, ti]);
      matchedDets.add(di);
      matchedTrks.add(ti);
    }
  }

  const unmatchedDets = detBboxes.map((_, i) => i).filter(i => !matchedDets.has(i));
  const unmatchedTrks = trkBboxes.map((_, i) => i).filter(i => !matchedTrks.has(i));

  return { matches: matched, unmatchedDets, unmatchedTrks };
}

export class ByteTracker {
  private tracks: Track[] = [];

  /** Reset tracker state (e.g. per-job). */
  reset() {
    this.tracks = [];
    _nextTrackId = 1;
  }

  /**
   * Update tracker with a new frame's detections.
   * Returns the stabilised detections with globally consistent track IDs.
   */
  update(detections: RawDetection[]): TrackedDetection[] {
    if (detections.length === 0 && this.tracks.length === 0) return [];

    // Split tracks by state
    const activeTracks  = this.tracks.filter(t => t.state === "active");
    const lostTracks    = this.tracks.filter(t => t.state === "lost");

    // Predict positions for all tracks
    const activePredicted = activeTracks.map(predictBbox);
    const lostPredicted   = lostTracks.map(predictBbox);

    const detBboxes = detections.map(d => d.bbox);

    // ── Stage 1: Match high-confidence detections → active tracks ─────────────
    const highDetIdx  = detections.map((_, i) => i).filter(i => detections[i]!.confidence >= 0.5);
    const highBboxes  = highDetIdx.map(i => detBboxes[i]!);

    const stage1 = greedyMatch(highBboxes, activePredicted, IOU_THRESHOLD_HIGH);

    const updatedActiveIds = new Set<number>();
    const usedDetIdxs      = new Set<number>();

    // Apply matched pairs
    for (const [localDi, ti] of stage1.matches) {
      const di   = highDetIdx[localDi]!;
      const det  = detections[di]!;
      const trk  = activeTracks[ti]!;

      const prevBbox = trk.bbox;
      trk.bbox      = det.bbox;
      trk.velocity  = [det.bbox[0] - prevBbox[0], det.bbox[1] - prevBbox[1]];
      trk.species   = det.species; // trust latest classification
      trk.confidence = det.confidence;
      trk.framesLost = 0;
      trk.hits++;
      trk.age++;
      trk.state = "active";

      updatedActiveIds.add(ti);
      usedDetIdxs.add(di);
    }

    // ── Stage 2: Remaining detections → lost tracks (re-ID) ──────────────────
    const remHighDet = stage1.unmatchedDets.map(i => highDetIdx[i]!).filter(i => !usedDetIdxs.has(i));
    const lowDetIdx  = detections.map((_, i) => i).filter(i => !usedDetIdxs.has(i) && detections[i]!.confidence < 0.5);
    const remDetIdx  = [...remHighDet, ...lowDetIdx];

    if (remDetIdx.length > 0 && lostTracks.length > 0) {
      const remBboxes = remDetIdx.map(i => detBboxes[i]!);
      const stage2    = greedyMatch(remBboxes, lostPredicted, IOU_THRESHOLD_LOW);

      for (const [localDi, ti] of stage2.matches) {
        const di   = remDetIdx[localDi]!;
        const det  = detections[di]!;
        const trk  = lostTracks[ti]!;

        trk.bbox       = det.bbox;
        trk.velocity   = [0, 0];
        trk.species    = det.species;
        trk.confidence = det.confidence;
        trk.framesLost = 0;
        trk.hits++;
        trk.age++;
        trk.state = "active";

        usedDetIdxs.add(di);
      }

      const stage2UnmatchedTrks = stage2.unmatchedTrks.map(i => lostTracks[i]!);
      for (const trk of stage2UnmatchedTrks) {
        trk.framesLost++;
        trk.age++;
        if (trk.framesLost > MAX_AGE) trk.state = "lost"; // will be pruned
      }
    } else {
      for (const trk of lostTracks) {
        trk.framesLost++;
        trk.age++;
      }
    }

    // Increment age for unmatched active tracks → move to lost
    for (let ti = 0; ti < activeTracks.length; ti++) {
      if (!updatedActiveIds.has(ti)) {
        const trk = activeTracks[ti]!;
        trk.framesLost++;
        trk.age++;
        if (trk.framesLost >= 1) trk.state = "lost";
      }
    }

    // ── Stage 3: Unmatched detections → new tracks ────────────────────────────
    for (const di of detections.map((_, i) => i)) {
      if (!usedDetIdxs.has(di)) {
        const det = detections[di]!;
        this.tracks.push({
          trackId:    _nextTrackId++,
          species:    det.species,
          bbox:       det.bbox,
          velocity:   [0, 0],
          confidence: det.confidence,
          age:        1,
          hits:       1,
          framesLost: 0,
          state:      "active",
        });
      }
    }

    // Prune dead tracks
    this.tracks = this.tracks.filter(
      t => !(t.state === "lost" && t.framesLost > MAX_AGE),
    );

    // Build output: all active tracks + recently-lost (for visual continuity)
    const output: TrackedDetection[] = this.tracks
      .filter(t => t.state === "active")
      .map(t => ({
        trackId:     t.trackId,
        species:     t.species,
        bbox:        t.bbox,
        confidence:  t.confidence,
        isConfirmed: t.hits >= MIN_HITS_TO_CONFIRM,
        framesLost:  t.framesLost,
      }));

    return output;
  }

  /** Return all track IDs seen so far grouped by species (for unique-bird counting). */
  getUniqueTracksBySpecies(): Map<string, Set<number>> {
    const map = new Map<string, Set<number>>();
    for (const trk of this.tracks) {
      if (!map.has(trk.species)) map.set(trk.species, new Set());
      map.get(trk.species)!.add(trk.trackId);
    }
    return map;
  }

  getAllTracks(): Track[] {
    return [...this.tracks];
  }
}
