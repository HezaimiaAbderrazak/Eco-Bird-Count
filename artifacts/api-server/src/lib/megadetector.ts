/**
 * MegaDetector integration — Stage 1 of the two-stage detection pipeline.
 *
 * Mirrors MegaDetector's core architecture:
 *   Stage 1 (this file): YOLO-based localization — WHERE are the birds?
 *   Stage 2 (analysis.ts): Gemini-based classification — WHAT species?
 *
 * The Python script (python/detector.py) runs YOLOv8n via ONNX Runtime,
 * using the same YOLO family of models as MegaDetector (v5/v8) but
 * pre-trained on COCO-80 which includes the "bird" class (class 14).
 *
 * MegaDetector paper reference:
 *   Beery et al. "Efficient Pipeline for Camera Trap Image Review" (2019)
 *   https://arxiv.org/abs/1907.06772
 */

import { spawn } from "child_process";
import { join } from "path";
import { fileURLToPath } from "url";
import { dirname } from "path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const DETECTOR_SCRIPT  = join(__dirname, "../../python/detector.py");
// Use the uv-managed venv that has ultralytics installed.
const PYTHON_BIN = join(__dirname, "../../../../.pythonlibs/bin/python3");

export interface YoloBirdDetection {
  bbox: [number, number, number, number]; // [x, y, w, h] normalised 0–1
  confidence: number;
  label: "bird" | "region"; // "bird" = YOLO confirmed; "region" = saliency proposal
}

export interface FrameDetectionResult {
  image: string;
  detections: YoloBirdDetection[];
}

/**
 * Run MegaDetector-style YOLO detection on a batch of frames.
 *
 * Spawns the Python detector in batch mode (all frames in one process call)
 * to amortise model-loading overhead — mirrors MegaDetector's batch API design.
 *
 * @param framePaths  Absolute paths to JPEG frames
 * @param timeoutMs   Per-batch timeout (default 120 s)
 * @returns           Per-frame list of bird bounding boxes
 */
export async function runMegaDetector(
  framePaths: string[],
  timeoutMs = 120_000,
): Promise<FrameDetectionResult[]> {
  if (framePaths.length === 0) return [];

  return new Promise((resolve, reject) => {
    const child = spawn(PYTHON_BIN, [DETECTOR_SCRIPT, ...framePaths], {
      env: {
        ...process.env,
        YOLO_MODEL_PATH: "yolov8n.pt",
      },
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (d: Buffer) => { stdout += d.toString(); });
    child.stderr.on("data", (d: Buffer) => {
      const msg = d.toString().trim();
      if (msg) console.log(`[MegaDetector] ${msg}`);
      stderr += msg;
    });

    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error(`MegaDetector timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    child.on("close", (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        console.error(`[MegaDetector] Process exited with code ${code}: ${stderr}`);
        // Graceful fallback — return empty detections so Gemini takes over
        resolve(framePaths.map(image => ({ image, detections: [] })));
        return;
      }

      try {
        const results: FrameDetectionResult[] = JSON.parse(stdout);
        resolve(results);
      } catch (err) {
        console.error("[MegaDetector] Failed to parse output:", stdout.slice(0, 200));
        resolve(framePaths.map(image => ({ image, detections: [] })));
      }
    });

    child.on("error", (err) => {
      clearTimeout(timer);
      console.error("[MegaDetector] Spawn error:", err.message);
      resolve(framePaths.map(image => ({ image, detections: [] })));
    });
  });
}

/**
 * Merge YOLO bounding boxes into a Gemini detection prompt context.
 *
 * Tells Gemini exactly WHERE the YOLO detector found birds so it can
 * focus classification on those regions — replicating MegaDetector's
 * two-stage "detect then classify" workflow.
 */
export function buildGuidedPrompt(detections: YoloBirdDetection[]): string {
  if (detections.length === 0) return "";

  const yoloBirds = detections.filter(d => d.label === "bird");
  const regions   = detections.filter(d => d.label === "region");

  const lines: string[] = [];

  if (yoloBirds.length > 0) {
    lines.push(`YOLO PRE-DETECTION (MegaDetector stage 1): ${yoloBirds.length} bird(s) confirmed:`);
    yoloBirds.forEach((d, i) => {
      const [x, y, w, h] = d.bbox;
      lines.push(`  YOLO Bird ${i + 1}: bbox=[x=${x.toFixed(3)}, y=${y.toFixed(3)}, w=${w.toFixed(3)}, h=${h.toFixed(3)}], confidence=${(d.confidence * 100).toFixed(0)}%`);
    });
    lines.push(`For each YOLO bird above, identify the exact species and return a detection using that bbox.`);
    lines.push(`You may adjust the bbox slightly if needed. Only add extra detections if >80% confident.`);
  }

  if (regions.length > 0) {
    lines.push(`\nSALIENCY REGIONS (high-activity zones likely to contain birds):`);
    regions.forEach((d, i) => {
      const [x, y, w, h] = d.bbox;
      lines.push(`  Region ${i + 1}: bbox=[x=${x.toFixed(3)}, y=${y.toFixed(3)}, w=${w.toFixed(3)}, h=${h.toFixed(3)}], activity=${(d.confidence * 100).toFixed(0)}%`);
    });
    lines.push(`Carefully examine each region above. If it contains birds, return a detection with the species.`);
    lines.push(`If a region contains no birds, skip it. You may also detect birds outside these regions.`);
  }

  return "\n\n" + lines.join("\n") + "\n";
}
