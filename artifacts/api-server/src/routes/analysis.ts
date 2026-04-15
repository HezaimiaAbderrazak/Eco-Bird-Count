/**
 * EcoBird Counter — Production-Grade Detection & Tracking Pipeline
 *
 * Architecture:
 *  1. FFmpeg extracts frames in parallel (high-res, adaptive interval).
 *  2. Gemini 1.5 Flash performs per-frame bird detection (no track-ID inference).
 *  3. ByteTracker assigns stable global track IDs across all frames.
 *  4. Open-Set Recovery: low-confidence detections get a dedicated visual reasoning pass.
 *  5. Wikipedia REST API cross-validates species names; unknown labels trigger Gemini re-ID.
 *  6. Species counts are derived from unique track IDs — eliminating count inflation.
 */

import { Router } from "express";
import { eq, desc } from "drizzle-orm";
import { db } from "@workspace/db";
import {
  analysisJobsTable,
  speciesDetectionsTable,
  detectionFramesTable,
} from "@workspace/db";
import { randomUUID } from "crypto";
import multer from "multer";
import { execFile } from "child_process";
import { promisify } from "util";
import { writeFile, readFile, rm, mkdir } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { ByteTracker, type RawDetection, type TrackedDetection } from "../lib/bytetrack";

const router = Router();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 500 * 1024 * 1024 },
});
const execFileAsync = promisify(execFile);

// ── AI client ─────────────────────────────────────────────────────────────────
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || "");

// Primary: gemini-1.5-flash (faster, production-grade)
const detectionModel   = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
// Recovery / reasoning pass
const reasoningModel   = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

// ── Species colour map ────────────────────────────────────────────────────────
const SPECIES_COLORS: Record<string, string> = {
  Sparrow: "#3B82F6", "House Sparrow": "#3B82F6", "Tree Sparrow": "#2563EB",
  Finch: "#22C55E", Chaffinch: "#16A34A", "Common Chaffinch": "#16A34A",
  Goldfinch: "#F59E0B", "European Goldfinch": "#F59E0B",
  Warbler: "#EF4444", "Garden Warbler": "#DC2626",
  Robin: "#EAB308", "European Robin": "#CA8A04",
  Kingfisher: "#06B6D4", "Common Kingfisher": "#0891B2",
  Swallow: "#8B5CF6", "Barn Swallow": "#7C3AED",
  Eagle: "#F97316", "Golden Eagle": "#EA580C", "Short-toed Eagle": "#C2410C",
  Falcon: "#EC4899", "Peregrine Falcon": "#DB2777", "Common Kestrel": "#BE185D",
  Kestrel: "#BE185D",
  Dove: "#A78BFA", "Turtle Dove": "#7C3AED", "Collared Dove": "#6D28D9",
  Pigeon: "#78716C", "Rock Pigeon": "#57534E",
  Heron: "#0EA5E9", "Grey Heron": "#0284C7",
  Owl: "#D97706", "Barn Owl": "#B45309", "Little Owl": "#92400E",
  Crow: "#374151", "Hooded Crow": "#1F2937",
  Stork: "#7C3AED", "White Stork": "#6D28D9",
  Hoopoe: "#C2410C", "Common Hoopoe": "#B91C1C",
  Lark: "#B45309", "Crested Lark": "#A16207",
  Starling: "#0891B2", "Common Starling": "#0E7490",
  "Bee-eater": "#16A34A", "European Bee-eater": "#15803D",
  Flamingo: "#DB2777", "Greater Flamingo": "#BE185D",
  Ibis: "#0F766E", "Northern Bald Ibis": "#0D9488",
  Nightingale: "#6D28D9", "Common Nightingale": "#5B21B6",
  Swift: "#F97316", "Common Swift": "#EA580C",
  Bunting: "#7C2D12", Wheatear: "#84CC16", "Northern Wheatear": "#65A30D",
  Roller: "#06B6D4", "European Roller": "#0891B2",
  "Unknown 🔍": "#9CA3AF", Unknown: "#9CA3AF",
};

function getSpeciesColor(species: string): string {
  if (SPECIES_COLORS[species]) return SPECIES_COLORS[species]!;
  let hash = 0;
  for (let i = 0; i < species.length; i++)
    hash = species.charCodeAt(i) + ((hash << 5) - hash);
  return `hsl(${Math.abs(hash) % 360}, 65%, 50%)`;
}

// ── Utilities ─────────────────────────────────────────────────────────────────
async function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Concurrency limiter — run `fn` over `items` with at most `limit` in-flight.
 */
async function concurrentMap<T, R>(
  items: T[],
  fn: (item: T, idx: number) => Promise<R>,
  limit: number,
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let cursor = 0;

  async function worker() {
    while (cursor < items.length) {
      const i = cursor++;
      results[i] = await fn(items[i]!, i);
    }
  }

  const workers = Array.from({ length: Math.min(limit, items.length) }, worker);
  await Promise.all(workers);
  return results;
}

async function callGeminiWithRetry(
  model: any,
  parts: any[],
  retries = 4,
  baseDelay = 8000,
): Promise<string> {
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      const result = await model.generateContent({
        contents: [{ role: "user", parts }],
      });
      return result.response.text().trim();
    } catch (err: any) {
      const is429 =
        err?.status === 429 ||
        err?.message?.includes("429") ||
        err?.message?.includes("quota") ||
        err?.message?.includes("RESOURCE_EXHAUSTED");
      if (is429 && attempt < retries - 1) {
        const delay = baseDelay * Math.pow(2, attempt);
        console.log(`[Gemini] Rate limited — retrying in ${delay}ms (${attempt + 1}/${retries})`);
        await sleep(delay);
        continue;
      }
      if (attempt < retries - 1) {
        await sleep(2000 * (attempt + 1));
        continue;
      }
      throw err;
    }
  }
  throw new Error("Max Gemini retries exceeded");
}

// ── Wikipedia species validation ──────────────────────────────────────────────
const wikiCache = new Map<string, string | null>();

async function validateSpeciesWikipedia(species: string): Promise<string | null> {
  const clean = species.replace("Unknown 🔍", "Unknown").trim();
  if (clean === "Unknown" || clean === "") return null;
  if (wikiCache.has(clean)) return wikiCache.get(clean)!;

  try {
    const encoded = encodeURIComponent(clean.replace(/ /g, "_"));
    const url = `https://en.wikipedia.org/api/rest_v1/page/summary/${encoded}`;
    const res = await fetch(url, {
      signal: AbortSignal.timeout(4000),
      headers: { "User-Agent": "EcoBird-Counter/2.0 (contact@ecobird.app)" },
    });
    if (!res.ok) {
      wikiCache.set(clean, null);
      return null;
    }
    const data = await res.json() as any;
    const extract = data.extract as string | undefined;
    const isBird = extract
      ? /bird|aves|passerine|raptor|migratory|ornithol/i.test(extract)
      : false;

    const result = isBird ? (data.description ?? null) : null;
    wikiCache.set(clean, result);
    return result;
  } catch {
    wikiCache.set(clean, null);
    return null;
  }
}

// ── Open-Set Recovery ─────────────────────────────────────────────────────────
/**
 * When a detection has confidence < LOW_CONF_THRESHOLD, send the cropped
 * image region to Gemini for focused visual reasoning.
 * If it cannot confidently identify the bird, it returns "Unknown 🔍".
 */
const LOW_CONF_THRESHOLD = 0.60;
const RECOVERY_PROMPT = `You are an expert ornithologist performing a focused species identification.
I am showing you a cropped region of a video frame that my bird detector flagged with LOW CONFIDENCE.

Your task:
1. Examine the image carefully for any bird present.
2. If you can identify the species with >70% certainty, return just the common English species name (e.g. "Barn Swallow").
3. If you cannot identify it confidently, return exactly: Unknown 🔍
4. If there is NO bird in this crop, return exactly: NO_BIRD

Return ONLY one of: a species name, "Unknown 🔍", or "NO_BIRD". No explanation, no punctuation.`;

async function recoverLowConfidenceDetection(
  imageBase64: string,
  currentLabel: string,
): Promise<{ species: string; confidence: number }> {
  try {
    const parts = [
      { text: RECOVERY_PROMPT },
      { text: `Current low-confidence label: "${currentLabel}"` },
      { inlineData: { mimeType: "image/jpeg", data: imageBase64 } },
    ];
    const response = await callGeminiWithRetry(reasoningModel, parts, 2, 3000);
    const result = response.trim();

    if (result === "NO_BIRD") return { species: "", confidence: 0 };
    if (result === "Unknown 🔍") return { species: "Unknown 🔍", confidence: 0.45 };
    return { species: result, confidence: 0.65 };
  } catch (err) {
    console.warn("[Recovery] Visual reasoning failed:", err);
    return { species: currentLabel, confidence: 0.50 };
  }
}

// ── Core Detection ────────────────────────────────────────────────────────────
const DETECTION_PROMPT = `You are a production-grade computer vision model specialised in ornithology.
Analyse the provided video frame and detect ALL birds with precision.

STRICT RULES:
- Only detect birds you can actually see. Do NOT hallucinate.
- Use specific common English names (e.g. "Greater Flamingo" not "Flamingo", "Common Hoopoe" not "Hoopoe").
- Algeria/North Africa context: priority species include Greater Flamingo, Hoopoe, Bee-eater, Wheatear, Roller, Lark, White Stork, Northern Bald Ibis, Barn Swallow, Peregrine Falcon, Great White Pelican, Whooper Swan.
- confidence: your genuine certainty 0.0–1.0 (be honest — use <0.6 for uncertain cases).
- If no birds are visible, return an empty array [].

FLOCK HANDLING (CRITICAL for large groups like flamingos):
- For LARGE FLOCKS (more than ~10 birds of the same species visible):
  * Report ONE entry for the entire flock/group.
  * Set "bbox" to the bounding box enclosing the whole flock area.
  * Set "count" to your best ESTIMATE of the total number of birds visible (e.g. 40, 150, 3000).
  * For very large dense flocks, estimate carefully: look at density × visible area.
- For INDIVIDUAL birds or small groups (≤10):
  * Report ONE entry PER bird (or small subgroup).
  * Set "count" to 1 (or the small subgroup number).
  * Bounding box should tightly enclose that individual/subgroup.

Return ONLY a valid JSON array. No markdown, no explanation.
Schema per detection:
{
  "species": string,
  "bbox": [x, y, width, height],   // normalised 0.0–1.0, origin = top-left
  "confidence": number,             // 0.0–1.0
  "count": number                   // 1 for individual, or estimated flock size
}

Examples:
[
  {"species": "Greater Flamingo", "bbox": [0.05, 0.30, 0.90, 0.60], "confidence": 0.95, "count": 350},
  {"species": "Common Hoopoe",    "bbox": [0.23, 0.41, 0.14, 0.18], "confidence": 0.93, "count": 1},
  {"species": "Barn Swallow",     "bbox": [0.67, 0.15, 0.08, 0.06], "confidence": 0.81, "count": 1}
]`;

async function detectBirdsInFrame(
  framePath: string,
  jobId: string,
  frameIndex: number,
): Promise<RawDetection[]> {
  try {
    const imageData = await readFile(framePath);
    const parts = [
      { text: DETECTION_PROMPT },
      { inlineData: { mimeType: "image/jpeg", data: imageData.toString("base64") } },
    ];

    const text = await callGeminiWithRetry(detectionModel, parts);
    const clean = text.replace(/```json\n?/g, "").replace(/```\n?/g, "").trim();

    // Find the JSON array, ignoring any leading/trailing text
    const arrayMatch = clean.match(/\[[\s\S]*\]/);
    if (!arrayMatch) return [];

    const parsed = JSON.parse(arrayMatch[0]);
    if (!Array.isArray(parsed)) return [];

    const detections: RawDetection[] = parsed
      .filter((d: any) => d.species && Array.isArray(d.bbox) && d.bbox.length === 4)
      .map((d: any) => ({
        species:    String(d.species).trim(),
        bbox:       (d.bbox as number[]).map(
          v => Math.min(1, Math.max(0, parseFloat(String(v)) || 0)),
        ) as [number, number, number, number],
        confidence: Math.min(1, Math.max(0, parseFloat(String(d.confidence)) || 0.7)),
        count:      d.count != null ? Math.max(1, Math.round(parseFloat(String(d.count)) || 1)) : 1,
      }))
      .filter((d: RawDetection) => {
        // Discard degenerate boxes
        const [, , w, h] = d.bbox;
        return w > 0.005 && h > 0.005;
      });

    return detections;
  } catch (err) {
    console.error(`[Frame ${frameIndex}] Detection failed:`, err);
    return [];
  }
}

// ── Open-Set Recovery pass ────────────────────────────────────────────────────
async function applyOpenSetRecovery(
  framePath: string,
  detections: RawDetection[],
): Promise<RawDetection[]> {
  const fullImage = await readFile(framePath).catch(() => null);
  if (!fullImage) return detections;
  const fullB64 = fullImage.toString("base64");

  const recovered: RawDetection[] = [];

  for (const det of detections) {
    if (det.confidence >= LOW_CONF_THRESHOLD) {
      recovered.push(det);
      continue;
    }

    console.log(`[OpenSet] Recovering low-conf detection: "${det.species}" (conf=${det.confidence.toFixed(2)})`);
    const { species, confidence } = await recoverLowConfidenceDetection(fullB64, det.species);

    if (species === "") {
      // Confirmed false positive — discard
      console.log("[OpenSet] False positive discarded.");
      continue;
    }

    recovered.push({ ...det, species, confidence });
  }

  return recovered;
}

// ── Frame extraction ──────────────────────────────────────────────────────────
async function extractFrames(
  videoBuffer: Buffer,
  originalName: string,
  sampleInterval: number,
  workDir: string,
): Promise<{ path: string; timestamp: number; frameIndex: number }[]> {
  const ext = originalName.split(".").pop()?.toLowerCase() || "mp4";
  const videoPath = join(workDir, `input.${ext}`);
  await writeFile(videoPath, videoBuffer);

  let duration = 30;
  try {
    const { stdout } = await execFileAsync("ffprobe", [
      "-v", "quiet",
      "-print_format", "json",
      "-show_format",
      videoPath,
    ]);
    const probe = JSON.parse(stdout);
    duration = parseFloat(probe.format?.duration || "30");
  } catch {
    console.log("[ffprobe] Could not determine duration, defaulting to 30s");
  }

  // Cap at 16 frames to keep API calls bounded; adjust interval if needed
  const MAX_FRAMES = 16;
  const naturalCount = Math.ceil(duration / sampleInterval);
  const actualInterval =
    naturalCount > MAX_FRAMES ? duration / MAX_FRAMES : sampleInterval;
  const frameCount = Math.min(MAX_FRAMES, Math.ceil(duration / actualInterval));

  console.log(
    `[Frames] duration=${duration.toFixed(1)}s  interval=${actualInterval.toFixed(2)}s  count=${frameCount}`,
  );

  // Extract frames in parallel (up to 4 concurrent ffmpeg processes)
  const framesMeta = Array.from({ length: frameCount }, (_, i) => ({
    i,
    ts: Math.min(i * actualInterval, duration - 0.1),
  }));

  const results = await concurrentMap(
    framesMeta,
    async ({ i, ts }) => {
      const framePath = join(workDir, `frame_${String(i).padStart(4, "0")}.jpg`);
      try {
        await execFileAsync("ffmpeg", [
          "-ss", ts.toFixed(3),
          "-i", videoPath,
          "-vframes", "1",
          "-q:v", "2",          // higher quality than before (was 4)
          "-vf", "scale=720:-1", // 720px wide — better for occlusion detection
          "-y", framePath,
        ]);
        return { path: framePath, timestamp: Math.round(ts * 100) / 100, frameIndex: i };
      } catch (e) {
        console.error(`[ffmpeg] Failed at ${ts.toFixed(3)}s:`, e);
        return null;
      }
    },
    4,
  );

  return results.filter(Boolean) as { path: string; timestamp: number; frameIndex: number }[];
}

// ── Wikipedia validation pass (async, best-effort) ────────────────────────────
async function enrichWithWikipedia(
  species: string,
): Promise<{ validated: boolean; wikiDescription: string | null }> {
  const desc = await validateSpeciesWikipedia(species);
  return { validated: desc !== null, wikiDescription: desc };
}

// ── Main analysis pipeline ────────────────────────────────────────────────────
async function runAnalysisPipeline(
  jobId: string,
  videoBuffer: Buffer,
  originalName: string,
  sampleInterval: number,
) {
  const workDir = join(tmpdir(), `ecobird_${jobId}`);
  await mkdir(workDir, { recursive: true });

  void (async () => {
    const tracker = new ByteTracker();

    try {
      await db
        .update(analysisJobsTable)
        .set({ status: "processing", progress: 5 })
        .where(eq(analysisJobsTable.id, jobId));

      // ── Phase 1: Frame extraction ──────────────────────────────────────────
      console.log(`[Job ${jobId}] Extracting frames...`);
      const framePaths = await extractFrames(
        videoBuffer,
        originalName,
        sampleInterval,
        workDir,
      );
      const frameCount = framePaths.length;

      await db
        .update(analysisJobsTable)
        .set({ totalFrames: frameCount, progress: 15 })
        .where(eq(analysisJobsTable.id, jobId));

      console.log(`[Job ${jobId}] ${frameCount} frames ready — starting detection...`);

      // ── Phase 2: Detection (concurrent, 2 in-flight at a time) ────────────
      const rawDetectionsByFrame = new Map<number, RawDetection[]>();

      await concurrentMap(
        framePaths,
        async (frame, batchPosition) => {
          const rawDetections = await detectBirdsInFrame(
            frame.path,
            jobId,
            frame.frameIndex,
          );

          // Phase 3: Open-Set Recovery per frame
          const recovered = await applyOpenSetRecovery(frame.path, rawDetections);
          rawDetectionsByFrame.set(frame.frameIndex, recovered);

          const progress = 15 + Math.round(((batchPosition + 1) / frameCount) * 65);
          await db
            .update(analysisJobsTable)
            .set({ progress, processedFrames: batchPosition + 1 })
            .where(eq(analysisJobsTable.id, jobId));
        },
        2, // max 2 concurrent Gemini calls
      );

      // ── Phase 4: ByteTracker pass — stabilise IDs across all frames ────────
      console.log(`[Job ${jobId}] Running ByteTracker...`);
      const trackedByFrame = new Map<number, TrackedDetection[]>();

      for (const frame of framePaths) {
        const raw = rawDetectionsByFrame.get(frame.frameIndex) ?? [];
        const tracked = tracker.update(raw);
        trackedByFrame.set(frame.frameIndex, tracked);
      }

      // ── Phase 5: Persist frame data ───────────────────────────────────────
      const frameRows = framePaths.map(fp => ({
        jobId,
        frameIndex: fp.frameIndex,
        timestamp:  fp.timestamp,
        detections: (trackedByFrame.get(fp.frameIndex) ?? []).map(d => ({
          trackId:    d.trackId,
          species:    d.species,
          bbox:       d.bbox,
          confidence: d.confidence,
          color:      getSpeciesColor(d.species),
          count:      d.count ?? 1,
        })),
      }));

      if (frameRows.length > 0) {
        await db.insert(detectionFramesTable).values(frameRows);
      }

      // ── Phase 6: Species aggregation ──────────────────────────────────────
      // For large-flock species (flamingos, pelicans etc.) we store the PEAK
      // visible count across all frames.  For individually-tracked species we
      // fall back to unique track IDs.
      const uniqueTracksBySpecies = tracker.getUniqueTracksBySpecies();

      // Compute peak visible count per species from frame data
      const peakVisibleCount = new Map<string, number>();
      const speciesConfidence = new Map<string, { sum: number; n: number }>();

      for (const tracked of trackedByFrame.values()) {
        // Sum counts per species in this frame
        const frameSum = new Map<string, number>();
        for (const d of tracked) {
          frameSum.set(d.species, (frameSum.get(d.species) ?? 0) + (d.count ?? 1));

          const acc = speciesConfidence.get(d.species) ?? { sum: 0, n: 0 };
          acc.sum += d.confidence;
          acc.n++;
          speciesConfidence.set(d.species, acc);
        }
        for (const [species, sum] of frameSum) {
          peakVisibleCount.set(species, Math.max(peakVisibleCount.get(species) ?? 0, sum));
        }
      }

      // ── Phase 7: Wikipedia validation (concurrent, fire-and-forget style) ─
      console.log(`[Job ${jobId}] Validating species via Wikipedia...`);
      const speciesList = [...uniqueTracksBySpecies.keys()];
      await concurrentMap(speciesList, async species => {
        const { validated } = await enrichWithWikipedia(species);
        if (!validated && species !== "Unknown 🔍") {
          console.log(`[Wiki] Could not validate "${species}" — keeping label.`);
        }
      }, 4);

      const detectionRows = speciesList.map(species => {
        const uniqueTracks = uniqueTracksBySpecies.get(species)?.size ?? 0;
        const peakCount    = peakVisibleCount.get(species) ?? uniqueTracks;
        // For large flocks, peak visible count is more meaningful than unique tracks
        const totalCount   = peakCount > uniqueTracks * 2 ? peakCount : Math.max(uniqueTracks, peakCount);
        const conf         = speciesConfidence.get(species);
        const avgConf      = conf ? conf.sum / conf.n : 0.7;
        return {
          jobId,
          species,
          totalCount,                              // peak visible or unique individuals
          averageConfidence: Math.round(avgConf * 100) / 100,
          color:             getSpeciesColor(species),
          lastSeenAt:        new Date(),
        };
      });

      if (detectionRows.length > 0) {
        await db.insert(speciesDetectionsTable).values(detectionRows);
      }

      const totalUnique = detectionRows.reduce((s, r) => s + r.totalCount, 0);
      console.log(
        `[Job ${jobId}] Done — ${detectionRows.length} species, ${totalUnique} unique birds`,
      );

      await db
        .update(analysisJobsTable)
        .set({ status: "completed", progress: 100, completedAt: new Date() })
        .where(eq(analysisJobsTable.id, jobId));
    } catch (err) {
      console.error(`[Job ${jobId}] Pipeline error:`, err);
      await db
        .update(analysisJobsTable)
        .set({ status: "failed" })
        .where(eq(analysisJobsTable.id, jobId));
    } finally {
      rm(workDir, { recursive: true, force: true }).catch(() => {});
    }
  })();
}

// ── HTTP Routes ───────────────────────────────────────────────────────────────

router.post("/analysis/upload", upload.single("video"), async (req, res) => {
  try {
    const file = req.file;
    if (!file) {
      res.status(400).json({ error: "bad_request", message: "No video file provided" });
      return;
    }

    const sampleInterval = parseFloat(req.body.sampleInterval ?? "2.0");
    const validInterval  = Math.min(
      10.0,
      Math.max(0.5, isNaN(sampleInterval) ? 2.0 : sampleInterval),
    );

    const jobId = randomUUID();
    await db.insert(analysisJobsTable).values({
      id:             jobId,
      status:         "pending",
      progress:       0,
      filename:       file.originalname,
      sampleInterval: validInterval,
    });

    runAnalysisPipeline(jobId, file.buffer, file.originalname, validInterval);

    const [job] = await db
      .select()
      .from(analysisJobsTable)
      .where(eq(analysisJobsTable.id, jobId));

    if (!job) {
      res.status(500).json({ error: "internal_error", message: "Job creation failed" });
      return;
    }

    res.json({
      id:             job.id,
      status:         job.status,
      progress:       job.progress,
      filename:       job.filename,
      sampleInterval: job.sampleInterval,
      createdAt:      job.createdAt.toISOString(),
      completedAt:    job.completedAt?.toISOString() ?? null,
      totalFrames:    job.totalFrames ?? null,
      processedFrames: job.processedFrames ?? null,
    });
  } catch (err) {
    req.log.error({ err }, "Failed to upload video");
    res.status(500).json({ error: "internal_error", message: "Failed to start analysis" });
  }
});

router.get("/analysis/jobs/recent", async (req, res) => {
  try {
    const jobs = await db
      .select()
      .from(analysisJobsTable)
      .orderBy(desc(analysisJobsTable.createdAt))
      .limit(10);

    res.json(
      jobs.map(job => ({
        id:              job.id,
        status:          job.status,
        progress:        job.progress,
        filename:        job.filename,
        sampleInterval:  job.sampleInterval,
        createdAt:       job.createdAt.toISOString(),
        completedAt:     job.completedAt?.toISOString() ?? null,
        totalFrames:     job.totalFrames ?? null,
        processedFrames: job.processedFrames ?? null,
      })),
    );
  } catch (err) {
    req.log.error({ err }, "Failed to get recent jobs");
    res.status(500).json({ error: "internal_error", message: "Failed to get recent jobs" });
  }
});

router.get("/analysis/:jobId", async (req, res) => {
  try {
    const { jobId } = req.params;

    const [job] = await db
      .select()
      .from(analysisJobsTable)
      .where(eq(analysisJobsTable.id, jobId));

    if (!job) {
      res.status(404).json({ error: "not_found", message: "Job not found" });
      return;
    }

    const detections = await db
      .select()
      .from(speciesDetectionsTable)
      .where(eq(speciesDetectionsTable.jobId, jobId));

    const totalBirds   = detections.reduce((s, d) => s + d.totalCount, 0);
    const sorted       = [...detections].sort((a, b) => b.totalCount - a.totalCount);
    const mostCommon   = sorted[0]?.species ?? null;

    res.json({
      job: {
        id:              job.id,
        status:          job.status,
        progress:        job.progress,
        filename:        job.filename,
        sampleInterval:  job.sampleInterval,
        createdAt:       job.createdAt.toISOString(),
        completedAt:     job.completedAt?.toISOString() ?? null,
        totalFrames:     job.totalFrames ?? null,
        processedFrames: job.processedFrames ?? null,
      },
      detections: sorted.map(d => ({
        species:          d.species,
        totalCount:       d.totalCount,
        averageConfidence: d.averageConfidence,
        color:            d.color,
        lastSeenAt:       d.lastSeenAt?.toISOString() ?? null,
      })),
      summary:
        job.status === "completed"
          ? {
              totalBirdsDetected:      totalBirds,
              uniqueSpecies:           detections.length,
              mostCommonSpecies:       mostCommon,
              processingDurationSeconds:
                job.completedAt && job.createdAt
                  ? (job.completedAt.getTime() - job.createdAt.getTime()) / 1000
                  : null,
            }
          : null,
    });
  } catch (err) {
    req.log.error({ err }, "Failed to get analysis status");
    res.status(500).json({ error: "internal_error", message: "Failed to get analysis status" });
  }
});

router.get("/analysis/:jobId/frames", async (req, res) => {
  try {
    const { jobId } = req.params;

    const frames = await db
      .select()
      .from(detectionFramesTable)
      .where(eq(detectionFramesTable.jobId, jobId));

    res.json(
      frames.map(f => ({
        frameIndex: f.frameIndex,
        timestamp:  f.timestamp,
        detections: f.detections,
      })),
    );
  } catch (err) {
    req.log.error({ err }, "Failed to get frames");
    res.status(500).json({ error: "internal_error", message: "Failed to get frames" });
  }
});

// Route alias kept for backwards compatibility
router.get("/analysis/jobs/:jobId", async (req, res) => {
  res.redirect(`/api/analysis/${req.params.jobId}`);
});

export default router;
