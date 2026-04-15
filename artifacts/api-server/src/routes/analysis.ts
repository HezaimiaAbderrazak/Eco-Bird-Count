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
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

const FIXED_COLORS: Record<string, string> = {
  Sparrow: "#3B82F6",
  Finch: "#22C55E",
  Warbler: "#EF4444",
  Robin: "#EAB308",
  Kingfisher: "#06B6D4",
  Swallow: "#8B5CF6",
  Eagle: "#F97316",
  Unknown: "#9CA3AF",
  Falcon: "#EC4899",
  Dove: "#A78BFA",
  Pigeon: "#78716C",
  Heron: "#0EA5E9",
  Owl: "#D97706",
  Crow: "#374151",
  Hawk: "#DC2626",
  Stork: "#7C3AED",
  Kite: "#059669",
  Starling: "#0891B2",
  Hoopoe: "#C2410C",
  Lark: "#B45309",
  Bee_eater: "#16A34A",
  "Bee-eater": "#16A34A",
  Bunting: "#7C2D12",
  Nightingale: "#6D28D9",
  Ibis: "#0F766E",
  Flamingo: "#DB2777",
  Pelican: "#1D4ED8",
};

function getSpeciesColor(species: string): string {
  if (FIXED_COLORS[species]) return FIXED_COLORS[species];
  let hash = 0;
  for (let i = 0; i < species.length; i++) hash = species.charCodeAt(i) + ((hash << 5) - hash);
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue}, 65%, 50%)`;
}

const BIRD_DETECTION_PROMPT = `You are an expert ornithologist and computer vision system analyzing a video frame.

Carefully examine this image and identify ALL birds visible.

For EACH bird detected, provide:
1. The common English species name (be specific: "European Sparrow", "House Sparrow", "Common Finch", etc.)
2. Bounding box as normalized coordinates [x, y, width, height] where x,y is top-left corner, all values 0.0 to 1.0 relative to image dimensions
3. Confidence score from 0.0 to 1.0
4. A short unique trackId integer (1-99) for each distinct bird instance

Return ONLY a valid JSON object in this exact format, no markdown, no explanation:
{
  "birds": [
    {
      "species": "House Sparrow",
      "bbox": [0.2, 0.3, 0.15, 0.2],
      "confidence": 0.92,
      "trackId": 1
    }
  ],
  "scene_description": "brief description of the scene"
}

If NO birds are detected, return: {"birds": [], "scene_description": "no birds visible"}

Rules:
- Only include ACTUAL birds you can see, do not hallucinate
- Be specific about species when possible
- Bounding boxes must tightly enclose each bird
- Each bird in the scene gets a unique trackId`;

interface GeminiDetection {
  species: string;
  bbox: number[];
  confidence: number;
  trackId: number;
}

async function analyzeFrameWithGemini(imagePath: string): Promise<GeminiDetection[]> {
  try {
    const imageData = await readFile(imagePath);
    const base64Image = imageData.toString("base64");
    const mimeType = "image/jpeg";

    const result = await model.generateContent([
      { text: BIRD_DETECTION_PROMPT },
      { inlineData: { mimeType, data: base64Image } },
    ]);

    const text = result.response.text().trim();
    const cleanText = text.replace(/```json\n?/g, "").replace(/```\n?/g, "").trim();
    const parsed = JSON.parse(cleanText);

    if (!parsed.birds || !Array.isArray(parsed.birds)) return [];

    return parsed.birds
      .filter((b: any) => b.species && Array.isArray(b.bbox) && b.bbox.length === 4)
      .map((b: any, idx: number) => ({
        species: String(b.species).trim(),
        bbox: b.bbox.map((v: any) => Math.min(1, Math.max(0, parseFloat(v) || 0))),
        confidence: Math.min(1, Math.max(0, parseFloat(b.confidence) || 0.5)),
        trackId: parseInt(b.trackId) || idx + 1,
      }));
  } catch (err) {
    return [];
  }
}

async function extractFrames(videoBuffer: Buffer, sampleInterval: number, workDir: string): Promise<{ path: string; timestamp: number }[]> {
  const videoPath = join(workDir, "input.mp4");
  await writeFile(videoPath, videoBuffer);

  const probeResult = await execFileAsync("ffprobe", [
    "-v", "quiet",
    "-print_format", "json",
    "-show_format",
    videoPath,
  ]);
  const probe = JSON.parse(probeResult.stdout);
  const duration = parseFloat(probe.format?.duration || "30");

  const frameTimestamps: number[] = [];
  for (let t = 0; t < duration; t += sampleInterval) {
    frameTimestamps.push(Math.round(t * 100) / 100);
  }

  const framePaths: { path: string; timestamp: number }[] = [];
  for (const ts of frameTimestamps) {
    const framePath = join(workDir, `frame_${ts.toFixed(2).replace(".", "_")}.jpg`);
    try {
      await execFileAsync("ffmpeg", [
        "-ss", String(ts),
        "-i", videoPath,
        "-vframes", "1",
        "-q:v", "3",
        "-vf", "scale=640:-1",
        "-y",
        framePath,
      ]);
      framePaths.push({ path: framePath, timestamp: ts });
    } catch {
    }
  }

  return framePaths;
}

async function runGeminiAnalysis(jobId: string, videoBuffer: Buffer, sampleInterval: number) {
  const workDir = join(tmpdir(), `ecobird_${jobId}`);
  await mkdir(workDir, { recursive: true });

  void (async () => {
    try {
      await db.update(analysisJobsTable)
        .set({ status: "processing", progress: 0 })
        .where(eq(analysisJobsTable.id, jobId));

      const framePaths = await extractFrames(videoBuffer, sampleInterval, workDir);
      const frameCount = framePaths.length || 1;

      await db.update(analysisJobsTable)
        .set({ totalFrames: frameCount })
        .where(eq(analysisJobsTable.id, jobId));

      const trackMap = new Map<string, { count: number; totalConfidence: number }>();
      const frames = [];

      for (let i = 0; i < framePaths.length; i++) {
        const { path: framePath, timestamp } = framePaths[i]!;
        const detections = await analyzeFrameWithGemini(framePath);

        const frameDetections = detections.map(d => ({
          ...d,
          color: getSpeciesColor(d.species),
        }));

        for (const d of frameDetections) {
          const existing = trackMap.get(d.species) ?? { count: 0, totalConfidence: 0 };
          trackMap.set(d.species, {
            count: existing.count + 1,
            totalConfidence: existing.totalConfidence + d.confidence,
          });
        }

        frames.push({ jobId, frameIndex: i, timestamp, detections: frameDetections });

        const progress = Math.round(((i + 1) / frameCount) * 100);
        await db.update(analysisJobsTable)
          .set({ progress, processedFrames: i + 1 })
          .where(eq(analysisJobsTable.id, jobId));
      }

      if (frames.length > 0) {
        await db.insert(detectionFramesTable).values(frames);
      }

      const detections = Array.from(trackMap.entries()).map(([species, data]) => ({
        jobId,
        species,
        totalCount: data.count,
        averageConfidence: Math.round((data.totalConfidence / data.count) * 100) / 100,
        color: getSpeciesColor(species),
        lastSeenAt: new Date(),
      }));

      if (detections.length > 0) {
        await db.insert(speciesDetectionsTable).values(detections);
      }

      await db.update(analysisJobsTable)
        .set({ status: "completed", progress: 100, completedAt: new Date() })
        .where(eq(analysisJobsTable.id, jobId));
    } catch (err) {
      console.error("Gemini analysis failed:", err);
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

    const sampleInterval = parseFloat(req.body.sampleInterval ?? "1.5");
    const validInterval = Math.min(10.0, Math.max(0.5, isNaN(sampleInterval) ? 1.5 : sampleInterval));

    const jobId = randomUUID();
    await db.insert(analysisJobsTable).values({
      id: jobId,
      status: "pending",
      progress: 0,
      filename: file.originalname,
      sampleInterval: validInterval,
    });

    runGeminiAnalysis(jobId, file.buffer, validInterval);

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
