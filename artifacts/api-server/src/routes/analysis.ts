import { Router } from "express";
import { eq, desc } from "drizzle-orm";
import { db } from "@workspace/db";
import { analysisJobsTable, speciesDetectionsTable, detectionFramesTable } from "@workspace/db";
import { randomUUID } from "crypto";
import multer from "multer";

const router = Router();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 500 * 1024 * 1024 } });

const SPECIES_CONFIG: Record<string, { color: string }> = {
  Sparrow: { color: "#3B82F6" },
  Finch: { color: "#22C55E" },
  Warbler: { color: "#EF4444" },
  Robin: { color: "#EAB308" },
  Kingfisher: { color: "#06B6D4" },
  Swallow: { color: "#8B5CF6" },
  Eagle: { color: "#F97316" },
  Unknown: { color: "#9CA3AF" },
};

const SPECIES_LIST = Object.keys(SPECIES_CONFIG).filter(s => s !== "Unknown");

function simulateAnalysis(jobId: string, sampleInterval: number) {
  void (async () => {
    try {
      const frameCount = Math.max(10, Math.floor(60 / sampleInterval));
      await db.update(analysisJobsTable)
        .set({ status: "processing", totalFrames: frameCount })
        .where(eq(analysisJobsTable.id, jobId));

      const trackMap = new Map<string, { count: number; totalConfidence: number }>();
      const frames = [];

      for (let i = 0; i < frameCount; i++) {
        const timestamp = i * sampleInterval;
        const frameDetections: Array<{
          trackId: number;
          species: string;
          confidence: number;
          bbox: number[];
          color: string;
        }> = [];

        const numBirds = Math.floor(Math.random() * 4);
        for (let b = 0; b < numBirds; b++) {
          const isUnknown = Math.random() < 0.1;
          const species = isUnknown ? "Unknown" : SPECIES_LIST[Math.floor(Math.random() * SPECIES_LIST.length)]!;
          const confidence = isUnknown ? 0.4 + Math.random() * 0.2 : 0.65 + Math.random() * 0.35;
          const trackId = Math.floor(Math.random() * 20);
          const bbox = [
            Math.random() * 0.6,
            Math.random() * 0.6,
            0.1 + Math.random() * 0.2,
            0.1 + Math.random() * 0.2,
          ];

          const existing = trackMap.get(species) ?? { count: 0, totalConfidence: 0 };
          trackMap.set(species, {
            count: existing.count + 1,
            totalConfidence: existing.totalConfidence + confidence,
          });

          frameDetections.push({
            trackId,
            species,
            confidence: Math.round(confidence * 100) / 100,
            bbox,
            color: SPECIES_CONFIG[species]?.color ?? "#888888",
          });
        }

        frames.push({ jobId, frameIndex: i, timestamp, detections: frameDetections });

        const progress = Math.round(((i + 1) / frameCount) * 100);
        await db.update(analysisJobsTable)
          .set({ progress, processedFrames: i + 1 })
          .where(eq(analysisJobsTable.id, jobId));

        await new Promise(resolve => setTimeout(resolve, 300));
      }

      if (frames.length > 0) {
        await db.insert(detectionFramesTable).values(frames);
      }

      const detections = Array.from(trackMap.entries()).map(([species, data]) => ({
        jobId,
        species,
        totalCount: data.count,
        averageConfidence: Math.round((data.totalConfidence / data.count) * 100) / 100,
        color: SPECIES_CONFIG[species]?.color ?? "#888888",
        lastSeenAt: new Date(),
      }));

      if (detections.length > 0) {
        await db.insert(speciesDetectionsTable).values(detections);
      }

      await db.update(analysisJobsTable)
        .set({ status: "completed", progress: 100, completedAt: new Date() })
        .where(eq(analysisJobsTable.id, jobId));
    } catch (err) {
      await db.update(analysisJobsTable)
        .set({ status: "failed" })
        .where(eq(analysisJobsTable.id, jobId));
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

    const sampleInterval = parseFloat(req.body.sampleInterval ?? "1.0");
    const validInterval = Math.min(5.0, Math.max(0.5, isNaN(sampleInterval) ? 1.0 : sampleInterval));

    const jobId = randomUUID();
    await db.insert(analysisJobsTable).values({
      id: jobId,
      status: "pending",
      progress: 0,
      filename: file.originalname,
      sampleInterval: validInterval,
    });

    simulateAnalysis(jobId, validInterval);

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
