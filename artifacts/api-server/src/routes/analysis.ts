import { Router } from "express";
import { eq, desc } from "drizzle-orm";
import { db } from "@workspace/db";
import { analysisJobsTable, speciesDetectionsTable, detectionFramesTable } from "@workspace/db";
import { randomUUID } from "crypto";
import multer from "multer";
import { execFile } from "child_process";
import { promisify } from "util";
import { writeFile, readFile, rm, mkdir } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";
import { GoogleGenerativeAI } from "@google/generative-ai";

const router = Router();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 500 * 1024 * 1024 } });
const execFileAsync = promisify(execFile);

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || "");

const SPECIES_COLORS: Record<string, string> = {
  Sparrow: "#3B82F6",
  "House Sparrow": "#3B82F6",
  "Tree Sparrow": "#2563EB",
  Finch: "#22C55E",
  Chaffinch: "#16A34A",
  "Common Chaffinch": "#16A34A",
  Goldfinch: "#F59E0B",
  "European Goldfinch": "#F59E0B",
  Warbler: "#EF4444",
  "Garden Warbler": "#DC2626",
  Robin: "#EAB308",
  "European Robin": "#CA8A04",
  Kingfisher: "#06B6D4",
  "Common Kingfisher": "#0891B2",
  Swallow: "#8B5CF6",
  "Barn Swallow": "#7C3AED",
  Eagle: "#F97316",
  "Golden Eagle": "#EA580C",
  "Short-toed Eagle": "#C2410C",
  Falcon: "#EC4899",
  "Peregrine Falcon": "#DB2777",
  "Common Kestrel": "#BE185D",
  Kestrel: "#BE185D",
  Dove: "#A78BFA",
  "Turtle Dove": "#7C3AED",
  "Collared Dove": "#6D28D9",
  Pigeon: "#78716C",
  "Rock Pigeon": "#57534E",
  Heron: "#0EA5E9",
  "Grey Heron": "#0284C7",
  Owl: "#D97706",
  "Barn Owl": "#B45309",
  "Little Owl": "#92400E",
  Crow: "#374151",
  "Hooded Crow": "#1F2937",
  Stork: "#7C3AED",
  "White Stork": "#6D28D9",
  Hoopoe: "#C2410C",
  "Common Hoopoe": "#B91C1C",
  Lark: "#B45309",
  "Crested Lark": "#A16207",
  Starling: "#0891B2",
  "Common Starling": "#0E7490",
  "Bee-eater": "#16A34A",
  "European Bee-eater": "#15803D",
  Flamingo: "#DB2777",
  "Greater Flamingo": "#BE185D",
  Ibis: "#0F766E",
  "Northern Bald Ibis": "#0D9488",
  Nightingale: "#6D28D9",
  "Common Nightingale": "#5B21B6",
  Swift: "#F97316",
  "Common Swift": "#EA580C",
  Bunting: "#7C2D12",
  Wheatear: "#84CC16",
  "Northern Wheatear": "#65A30D",
  Roller: "#06B6D4",
  "European Roller": "#0891B2",
  Unknown: "#9CA3AF",
};

function getSpeciesColor(species: string): string {
  if (SPECIES_COLORS[species]) return SPECIES_COLORS[species];
  let hash = 0;
  for (let i = 0; i < species.length; i++) hash = species.charCodeAt(i) + ((hash << 5) - hash);
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue}, 65%, 50%)`;
}

async function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function callGeminiWithRetry(
  geminiModel: any,
  parts: any[],
  retries = 3,
  baseDelay = 10000
): Promise<string> {
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      const result = await geminiModel.generateContent({ contents: [{ role: "user", parts }] });
      return result.response.text().trim();
    } catch (err: any) {
      const is429 = err?.status === 429 || err?.message?.includes("429") || err?.message?.includes("quota");
      if (is429 && attempt < retries - 1) {
        const delay = baseDelay * Math.pow(2, attempt);
        console.log(`[Gemini] Rate limited, waiting ${delay}ms before retry ${attempt + 1}/${retries}`);
        await sleep(delay);
        continue;
      }
      throw err;
    }
  }
  throw new Error("Max retries exceeded");
}

interface FrameDetection {
  trackId: number;
  species: string;
  confidence: number;
  bbox: number[];
  color: string;
}

const BATCH_ANALYSIS_PROMPT = `You are an expert ornithologist with computer vision expertise analyzing bird footage.

I will send you multiple video frames. For each frame, identify ALL birds visible.

Return a JSON object where each key is the frame index (0, 1, 2...) and the value is an array of bird detections.

For each bird detection provide:
- "species": Common English name, be specific (e.g. "House Sparrow", "Barn Swallow", "European Bee-eater")
- "bbox": [x, y, width, height] normalized coordinates 0.0-1.0 relative to image dimensions, tight around the bird
- "confidence": 0.0-1.0 detection confidence  
- "trackId": integer 1-99, unique per distinct bird instance

Example output format:
{
  "0": [{"species": "House Sparrow", "bbox": [0.2, 0.3, 0.12, 0.15], "confidence": 0.91, "trackId": 1}],
  "1": [],
  "2": [{"species": "Barn Swallow", "bbox": [0.5, 0.2, 0.18, 0.1], "confidence": 0.87, "trackId": 2}]
}

IMPORTANT RULES:
- Only detect birds actually visible, do not hallucinate
- If a frame has no birds, use an empty array []
- Bounding boxes must tightly enclose each bird
- Be specific with species names - prefer common name over generic (e.g. "Common Kestrel" not "bird of prey")
- Algeria/North Africa context: look for typical species like Hoopoe, Bee-eater, Wheatear, Lark, Roller, Stork

Return ONLY valid JSON, no markdown, no explanation.`;

async function analyzeBatchWithGemini(
  framePaths: { path: string; timestamp: number; frameIndex: number }[],
  geminiModel: any
): Promise<Map<number, FrameDetection[]>> {
  const results = new Map<number, FrameDetection[]>();

  const parts: any[] = [{ text: BATCH_ANALYSIS_PROMPT }];
  const validFrames: { path: string; timestamp: number; frameIndex: number }[] = [];

  for (const frame of framePaths) {
    try {
      const imageData = await readFile(frame.path);
      parts.push({ text: `\n--- Frame ${frame.frameIndex} (t=${frame.timestamp}s) ---` });
      parts.push({ inlineData: { mimeType: "image/jpeg", data: imageData.toString("base64") } });
      validFrames.push(frame);
    } catch {
      // skip unreadable frames
    }
  }

  if (validFrames.length === 0) return results;

  try {
    const text = await callGeminiWithRetry(geminiModel, [{ role: "user", parts }]);
    const cleanText = text.replace(/```json\n?/g, "").replace(/```\n?/g, "").trim();
    const parsed = JSON.parse(cleanText);

    for (const frame of validFrames) {
      const key = String(frame.frameIndex);
      const birds = parsed[key];
      if (!Array.isArray(birds)) {
        results.set(frame.frameIndex, []);
        continue;
      }

      const detections: FrameDetection[] = birds
        .filter((b: any) => b.species && Array.isArray(b.bbox) && b.bbox.length === 4)
        .map((b: any, idx: number) => ({
          species: String(b.species).trim(),
          bbox: (b.bbox as number[]).map(v => Math.min(1, Math.max(0, parseFloat(String(v)) || 0))),
          confidence: Math.min(1, Math.max(0, parseFloat(String(b.confidence)) || 0.75)),
          trackId: parseInt(String(b.trackId)) || idx + 1,
          color: getSpeciesColor(String(b.species).trim()),
        }));
      results.set(frame.frameIndex, detections);
    }
  } catch (err) {
    console.error("[Gemini] Batch analysis failed:", err);
    for (const frame of validFrames) results.set(frame.frameIndex, []);
  }

  return results;
}

async function extractFrames(
  videoBuffer: Buffer,
  originalName: string,
  sampleInterval: number,
  workDir: string
): Promise<{ path: string; timestamp: number; frameIndex: number }[]> {
  const ext = originalName.split(".").pop()?.toLowerCase() || "mp4";
  const videoPath = join(workDir, `input.${ext}`);
  await writeFile(videoPath, videoBuffer);

  let duration = 30;
  try {
    const probeResult = await execFileAsync("ffprobe", [
      "-v", "quiet",
      "-print_format", "json",
      "-show_format",
      videoPath,
    ]);
    const probe = JSON.parse(probeResult.stdout);
    duration = parseFloat(probe.format?.duration || "30");
  } catch {
    console.log("[ffprobe] Could not get duration, defaulting to 30s");
  }

  // Limit to MAX 12 frames to avoid API quota issues
  const maxFrames = 12;
  const naturalCount = Math.ceil(duration / sampleInterval);
  const actualInterval = naturalCount > maxFrames ? duration / maxFrames : sampleInterval;
  const frameCount = Math.min(maxFrames, Math.ceil(duration / actualInterval));

  const framePaths: { path: string; timestamp: number; frameIndex: number }[] = [];

  for (let i = 0; i < frameCount; i++) {
    const ts = Math.min(i * actualInterval, duration - 0.1);
    const framePath = join(workDir, `frame_${String(i).padStart(3, "0")}.jpg`);
    try {
      await execFileAsync("ffmpeg", [
        "-ss", String(ts.toFixed(3)),
        "-i", videoPath,
        "-vframes", "1",
        "-q:v", "4",
        "-vf", "scale=512:-1",
        "-y",
        framePath,
      ]);
      framePaths.push({ path: framePath, timestamp: Math.round(ts * 100) / 100, frameIndex: i });
    } catch (e) {
      console.error(`[ffmpeg] Failed to extract frame at ${ts}s:`, e);
    }
  }

  return framePaths;
}

async function runGeminiAnalysis(jobId: string, videoBuffer: Buffer, originalName: string, sampleInterval: number) {
  const workDir = join(tmpdir(), `ecobird_${jobId}`);
  await mkdir(workDir, { recursive: true });

  void (async () => {
    try {
      await db.update(analysisJobsTable)
        .set({ status: "processing", progress: 5 })
        .where(eq(analysisJobsTable.id, jobId));

      console.log(`[Job ${jobId}] Starting frame extraction...`);
      const framePaths = await extractFrames(videoBuffer, originalName, sampleInterval, workDir);
      const frameCount = framePaths.length;

      console.log(`[Job ${jobId}] Extracted ${frameCount} frames, starting Gemini analysis...`);
      await db.update(analysisJobsTable)
        .set({ totalFrames: frameCount, progress: 15 })
        .where(eq(analysisJobsTable.id, jobId));

      const BATCH_SIZE = 4;
      const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash-lite" });

      const allResults = new Map<number, FrameDetection[]>();
      const batches = [];
      for (let i = 0; i < framePaths.length; i += BATCH_SIZE) {
        batches.push(framePaths.slice(i, i + BATCH_SIZE));
      }

      for (let batchIdx = 0; batchIdx < batches.length; batchIdx++) {
        const batch = batches[batchIdx]!;
        console.log(`[Job ${jobId}] Processing batch ${batchIdx + 1}/${batches.length} (${batch.length} frames)`);

        const batchResults = await analyzeBatchWithGemini(batch, model);
        batchResults.forEach((detections, frameIdx) => allResults.set(frameIdx, detections));

        const progress = 15 + Math.round(((batchIdx + 1) / batches.length) * 70);
        const processed = Math.min((batchIdx + 1) * BATCH_SIZE, frameCount);
        await db.update(analysisJobsTable)
          .set({ progress, processedFrames: processed })
          .where(eq(analysisJobsTable.id, jobId));

        // Respectful delay between batches
        if (batchIdx < batches.length - 1) await sleep(3000);
      }

      // Build DB records
      const frames = framePaths.map(fp => ({
        jobId,
        frameIndex: fp.frameIndex,
        timestamp: fp.timestamp,
        detections: allResults.get(fp.frameIndex) ?? [],
      }));

      if (frames.length > 0) {
        await db.insert(detectionFramesTable).values(frames);
      }

      // Aggregate species counts
      const trackMap = new Map<string, { count: number; totalConfidence: number }>();
      for (const detections of allResults.values()) {
        for (const d of detections) {
          const existing = trackMap.get(d.species) ?? { count: 0, totalConfidence: 0 };
          trackMap.set(d.species, {
            count: existing.count + 1,
            totalConfidence: existing.totalConfidence + d.confidence,
          });
        }
      }

      const detectionRows = Array.from(trackMap.entries()).map(([species, data]) => ({
        jobId,
        species,
        totalCount: data.count,
        averageConfidence: Math.round((data.totalConfidence / data.count) * 100) / 100,
        color: getSpeciesColor(species),
        lastSeenAt: new Date(),
      }));

      if (detectionRows.length > 0) {
        await db.insert(speciesDetectionsTable).values(detectionRows);
      }

      console.log(`[Job ${jobId}] Completed: ${detectionRows.length} species detected`);
      await db.update(analysisJobsTable)
        .set({ status: "completed", progress: 100, completedAt: new Date() })
        .where(eq(analysisJobsTable.id, jobId));
    } catch (err) {
      console.error(`[Job ${jobId}] Analysis failed:`, err);
      await db.update(analysisJobsTable)
        .set({ status: "failed" })
        .where(eq(analysisJobsTable.id, jobId));
    } finally {
      rm(workDir, { recursive: true, force: true }).catch(() => {});
    }
  })();
}

router.post("/analysis/upload", upload.single("video"), async (req, res) => {
  try {
    const file = req.file;
    if (!file) {
      res.status(400).json({ error: "bad_request", message: "No video file provided" });
      return;
    }

    const sampleInterval = parseFloat(req.body.sampleInterval ?? "2.0");
    const validInterval = Math.min(10.0, Math.max(0.5, isNaN(sampleInterval) ? 2.0 : sampleInterval));

    const jobId = randomUUID();
    await db.insert(analysisJobsTable).values({
      id: jobId,
      status: "pending",
      progress: 0,
      filename: file.originalname,
      sampleInterval: validInterval,
    });

    runGeminiAnalysis(jobId, file.buffer, file.originalname, validInterval);

    const [job] = await db.select().from(analysisJobsTable).where(eq(analysisJobsTable.id, jobId));
    if (!job) {
      res.status(500).json({ error: "internal_error", message: "Job creation failed" });
      return;
    }

    res.json({
      id: job.id,
      status: job.status,
      progress: job.progress,
      filename: job.filename,
      sampleInterval: job.sampleInterval,
      createdAt: job.createdAt.toISOString(),
      completedAt: job.completedAt?.toISOString() ?? null,
      totalFrames: job.totalFrames ?? null,
      processedFrames: job.processedFrames ?? null,
    });
  } catch (err) {
    req.log.error({ err }, "Failed to upload video");
    res.status(500).json({ error: "internal_error", message: "Failed to start analysis" });
  }
});

router.get("/analysis/jobs/recent", async (req, res) => {
  try {
    const jobs = await db.select().from(analysisJobsTable)
      .orderBy(desc(analysisJobsTable.createdAt))
      .limit(10);

    res.json(jobs.map(job => ({
      id: job.id,
      status: job.status,
      progress: job.progress,
      filename: job.filename,
      sampleInterval: job.sampleInterval,
      createdAt: job.createdAt.toISOString(),
      completedAt: job.completedAt?.toISOString() ?? null,
      totalFrames: job.totalFrames ?? null,
      processedFrames: job.processedFrames ?? null,
    })));
  } catch (err) {
    req.log.error({ err }, "Failed to get recent jobs");
    res.status(500).json({ error: "internal_error", message: "Failed to get recent jobs" });
  }
});

router.get("/analysis/:jobId", async (req, res) => {
  try {
    const { jobId } = req.params;

    const [job] = await db.select().from(analysisJobsTable).where(eq(analysisJobsTable.id, jobId));
    if (!job) {
      res.status(404).json({ error: "not_found", message: "Job not found" });
      return;
    }

    const detections = await db.select().from(speciesDetectionsTable).where(
      eq(speciesDetectionsTable.jobId, jobId)
    );

    const totalBirds = detections.reduce((sum, d) => sum + d.totalCount, 0);
    const sortedDetections = [...detections].sort((a, b) => b.totalCount - a.totalCount);
    const mostCommon = sortedDetections[0]?.species ?? null;

    res.json({
      job: {
        id: job.id,
        status: job.status,
        progress: job.progress,
        filename: job.filename,
        sampleInterval: job.sampleInterval,
        createdAt: job.createdAt.toISOString(),
        completedAt: job.completedAt?.toISOString() ?? null,
        totalFrames: job.totalFrames ?? null,
        processedFrames: job.processedFrames ?? null,
      },
      detections: sortedDetections.map(d => ({
        species: d.species,
        totalCount: d.totalCount,
        averageConfidence: d.averageConfidence,
        color: d.color,
        lastSeenAt: d.lastSeenAt?.toISOString() ?? null,
      })),
      summary: job.status === "completed" ? {
        totalBirdsDetected: totalBirds,
        uniqueSpecies: detections.length,
        mostCommonSpecies: mostCommon,
        processingDurationSeconds: job.completedAt && job.createdAt
          ? (job.completedAt.getTime() - job.createdAt.getTime()) / 1000
          : null,
      } : null,
    });
  } catch (err) {
    req.log.error({ err }, "Failed to get analysis status");
    res.status(500).json({ error: "internal_error", message: "Failed to get analysis status" });
  }
});

router.get("/analysis/:jobId/frames", async (req, res) => {
  try {
    const { jobId } = req.params;

    const frames = await db.select().from(detectionFramesTable).where(
      eq(detectionFramesTable.jobId, jobId)
    );

    res.json(frames.map(f => ({
      frameIndex: f.frameIndex,
      timestamp: f.timestamp,
      detections: f.detections,
    })));
  } catch (err) {
    req.log.error({ err }, "Failed to get frames");
    res.status(500).json({ error: "internal_error", message: "Failed to get frames" });
  }
});

export default router;
