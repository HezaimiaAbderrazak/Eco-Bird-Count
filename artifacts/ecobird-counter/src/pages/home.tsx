import { useState, useRef, useEffect, useCallback } from "react";
import {
  UploadCloud, PlaySquare, Bird, AlertCircle, Info,
  Play, Pause, RotateCcw, ChevronRight, Loader2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import {
  useUploadVideo,
  useGetAnalysisStatus,
  getGetAnalysisStatusQueryKey,
  useGetAnalysisFrames,
  getGetAnalysisFramesQueryKey,
  useGetSpeciesInfo,
  getGetSpeciesInfoQueryKey,
} from "@workspace/api-client-react";
import { getSpeciesColor } from "@/lib/colors";
import { cn } from "@/lib/utils";

// ── Types ─────────────────────────────────────────────────────────────────────

interface BirdDetection {
  trackId: number;
  species: string;
  confidence: number;
  bbox: number[];
  color: string;
}

interface DetectionFrame {
  frameIndex: number;
  timestamp: number;
  detections: BirdDetection[];
}

// ── Simple video preview (no bounding boxes) ──────────────────────────────────

function VideoPreview({
  file,
  autoPlay,
  onReady,
}: {
  file: File;
  autoPlay: boolean;
  onReady?: (el: HTMLVideoElement) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const objectUrl = useRef("");

  useEffect(() => {
    objectUrl.current = URL.createObjectURL(file);
    return () => URL.revokeObjectURL(objectUrl.current);
  }, [file]);

  useEffect(() => {
    if (autoPlay && videoRef.current) {
      videoRef.current.play().catch(() => {});
      setIsPlaying(true);
    }
  }, [autoPlay]);

  const fmt = (s: number) =>
    `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

  const toggle = () => {
    if (!videoRef.current) return;
    if (videoRef.current.paused) {
      videoRef.current.play();
      setIsPlaying(true);
    } else {
      videoRef.current.pause();
      setIsPlaying(false);
    }
  };

  return (
    <div className="space-y-3">
      <div
        className="relative rounded-xl overflow-hidden bg-black shadow-lg border border-border"
        style={{ aspectRatio: "16/9" }}
      >
        <video
          ref={videoRef}
          src={objectUrl.current}
          className="w-full h-full object-contain"
          onTimeUpdate={() => setCurrentTime(videoRef.current?.currentTime ?? 0)}
          onLoadedMetadata={() => {
            const v = videoRef.current;
            if (v) { setDuration(v.duration); onReady?.(v); }
          }}
          onEnded={() => setIsPlaying(false)}
          playsInline
        />
        <div className="absolute bottom-3 right-3 bg-black/60 text-white text-xs font-mono px-2 py-1 rounded">
          {fmt(currentTime)} / {fmt(duration || 0)}
        </div>
      </div>
      <div className="flex items-center gap-3">
        <Button size="sm" variant="outline" onClick={toggle} className="gap-1.5 shrink-0">
          {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          {isPlaying ? "Pause" : "Play"}
        </Button>
        <Button size="sm" variant="ghost" onClick={() => {
          if (videoRef.current) { videoRef.current.currentTime = 0; videoRef.current.play(); setIsPlaying(true); }
        }} className="shrink-0 px-2">
          <RotateCcw className="h-4 w-4" />
        </Button>
        <Slider
          value={[currentTime]}
          onValueChange={([v]) => { if (videoRef.current) { videoRef.current.currentTime = v; setCurrentTime(v); } }}
          min={0}
          max={duration || 100}
          step={0.1}
          className="flex-1"
        />
      </div>
    </div>
  );
}

// ── Full player with bounding-box overlay ─────────────────────────────────────

function VideoPlayer({
  file,
  frames,
  onSpeciesClick,
  selectedSpecies,
}: {
  file: File;
  frames: DetectionFrame[];
  onSpeciesClick: (species: string) => void;
  selectedSpecies: string | null;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const objectUrl = useRef("");
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [liveCounters, setLiveCounters] = useState<Record<string, { count: number; color: string }>>({});
  const [currentDetections, setCurrentDetections] = useState<BirdDetection[]>([]);

  useEffect(() => {
    objectUrl.current = URL.createObjectURL(file);
    return () => URL.revokeObjectURL(objectUrl.current);
  }, [file]);

  const getCurrentFrame = useCallback(
    (time: number): DetectionFrame | null => {
      if (!frames.length) return null;
      let best: DetectionFrame | null = null;
      let minDiff = Infinity;
      for (const f of frames) {
        const diff = Math.abs(f.timestamp - time);
        if (diff < minDiff) { minDiff = diff; best = f; }
      }
      return best;
    },
    [frames],
  );

  const drawFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return;

    const rect = video.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const t = video.currentTime;
    const frame = getCurrentFrame(t);
    if (!frame || !frame.detections.length) { setCurrentDetections([]); return; }

    setCurrentDetections(frame.detections);

    // Live unique-track counters up to current time
    const seenKeys = new Set<string>();
    const counters: Record<string, { count: number; color: string }> = {};
    for (const f of frames) {
      if (f.timestamp > t + 0.5) break;
      for (const d of f.detections) {
        const key = `${d.species}-${d.trackId}`;
        if (!seenKeys.has(key)) {
          seenKeys.add(key);
          if (!counters[d.species]) counters[d.species] = { count: 0, color: d.color };
          counters[d.species]!.count++;
        }
      }
    }
    setLiveCounters(counters);

    for (const det of frame.detections) {
      const [x, y, w, h] = det.bbox;
      const px = x * canvas.width;
      const py = y * canvas.height;
      const pw = w * canvas.width;
      const ph = h * canvas.height;
      const color = det.color || getSpeciesColor(det.species);

      ctx.shadowColor = color;
      ctx.shadowBlur = 14;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(px, py, pw, ph);
      ctx.shadowBlur = 0;

      const cLen = Math.min(pw, ph) * 0.25;
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(px, py + cLen); ctx.lineTo(px, py); ctx.lineTo(px + cLen, py); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(px + pw - cLen, py); ctx.lineTo(px + pw, py); ctx.lineTo(px + pw, py + cLen); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(px, py + ph - cLen); ctx.lineTo(px, py + ph); ctx.lineTo(px + cLen, py + ph); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(px + pw - cLen, py + ph); ctx.lineTo(px + pw, py + ph); ctx.lineTo(px + pw, py + ph - cLen); ctx.stroke();

      const label = `#${det.trackId} ${det.species} ${Math.round(det.confidence * 100)}%`;
      const fontSize = Math.max(11, Math.min(14, canvas.width / 45));
      ctx.font = `bold ${fontSize}px 'Plus Jakarta Sans', sans-serif`;
      const tw = ctx.measureText(label).width;
      const th = fontSize + 6;
      const lx = px;
      const ly = py > th + 4 ? py - th - 4 : py + ph + 4;

      ctx.fillStyle = color;
      ctx.globalAlpha = 0.9;
      const br = 4;
      ctx.beginPath();
      ctx.moveTo(lx + br, ly);
      ctx.lineTo(lx + tw + 10 - br, ly);
      ctx.quadraticCurveTo(lx + tw + 10, ly, lx + tw + 10, ly + br);
      ctx.lineTo(lx + tw + 10, ly + th - br);
      ctx.quadraticCurveTo(lx + tw + 10, ly + th, lx + tw + 10 - br, ly + th);
      ctx.lineTo(lx + br, ly + th);
      ctx.quadraticCurveTo(lx, ly + th, lx, ly + th - br);
      ctx.lineTo(lx, ly + br);
      ctx.quadraticCurveTo(lx, ly, lx + br, ly);
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 1;

      ctx.fillStyle = "#fff";
      ctx.fillText(label, lx + 5, ly + th - 5);
    }
  }, [frames, getCurrentFrame]);

  useEffect(() => {
    let raf: number;
    const loop = () => { drawFrame(); raf = requestAnimationFrame(loop); };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [drawFrame]);

  const toggle = () => {
    if (!videoRef.current) return;
    if (videoRef.current.paused) { videoRef.current.play(); setIsPlaying(true); }
    else { videoRef.current.pause(); setIsPlaying(false); }
  };

  const fmt = (s: number) =>
    `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

  return (
    <div className="space-y-3">
      <div
        className="relative rounded-xl overflow-hidden bg-black shadow-lg border border-border"
        style={{ aspectRatio: "16/9" }}
      >
        <video
          ref={videoRef}
          src={objectUrl.current}
          className="w-full h-full object-contain"
          onTimeUpdate={() => setCurrentTime(videoRef.current?.currentTime ?? 0)}
          onLoadedMetadata={() => { if (videoRef.current) setDuration(videoRef.current.duration); }}
          onEnded={() => setIsPlaying(false)}
          playsInline
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
          style={{ zIndex: 10 }}
        />

        {/* Live species badge overlay */}
        {currentDetections.length > 0 && (
          <div className="absolute top-3 left-3 flex flex-wrap gap-1.5 z-20 max-w-[60%]">
            {currentDetections.map((d, i) => (
              <button
                key={`${d.trackId}-${i}`}
                onClick={() => onSpeciesClick(d.species)}
                className="flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-bold text-white shadow-lg transition-transform hover:scale-105"
                style={{ backgroundColor: d.color, opacity: 0.92 }}
              >
                <span className="w-1.5 h-1.5 rounded-full bg-white inline-block animate-pulse" />
                {d.species}
              </button>
            ))}
          </div>
        )}

        <div className="absolute top-3 right-3 z-20">
          <Badge className="bg-green-600 text-white text-xs">
            <span className="w-1.5 h-1.5 rounded-full bg-white mr-1.5 animate-pulse inline-block" />
            Live tracking
          </Badge>
        </div>

        <div className="absolute bottom-3 right-3 z-20 bg-black/60 text-white text-xs font-mono px-2 py-1 rounded">
          {fmt(currentTime)} / {fmt(duration || 0)}
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3">
        <Button size="sm" variant="outline" onClick={toggle} className="gap-1.5 shrink-0">
          {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          {isPlaying ? "Pause" : "Play"}
        </Button>
        <Button size="sm" variant="ghost" onClick={() => {
          if (videoRef.current) { videoRef.current.currentTime = 0; videoRef.current.play(); setIsPlaying(true); }
        }} className="shrink-0 px-2">
          <RotateCcw className="h-4 w-4" />
        </Button>
        <Slider
          value={[currentTime]}
          onValueChange={([v]) => { if (videoRef.current) { videoRef.current.currentTime = v; setCurrentTime(v); } }}
          min={0}
          max={duration || 100}
          step={0.1}
          className="flex-1"
        />
      </div>

      {/* Live unique-bird counters */}
      {Object.keys(liveCounters).length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2 mt-1">
          {Object.entries(liveCounters)
            .sort((a, b) => b[1].count - a[1].count)
            .map(([species, data]) => (
              <button
                key={species}
                onClick={() => onSpeciesClick(species)}
                className={cn(
                  "flex items-center gap-2 p-2.5 rounded-lg border transition-all text-left hover:shadow-sm",
                  selectedSpecies === species
                    ? "border-primary bg-primary/5 shadow-sm"
                    : "border-border bg-card hover:border-primary/40",
                )}
              >
                <div
                  className="h-7 w-7 rounded-full flex items-center justify-center text-white text-sm font-bold shrink-0"
                  style={{ backgroundColor: data.color }}
                >
                  {data.count}
                </div>
                <span className="text-xs font-medium truncate">{species}</span>
              </button>
            ))}
        </div>
      )}
    </div>
  );
}

// ── Home page ─────────────────────────────────────────────────────────────────

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [sampleInterval, setSampleInterval] = useState([1.5]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [selectedSpecies, setSelectedSpecies] = useState<string | null>(null);
  const [autoPlay, setAutoPlay] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const uploadMutation = useUploadVideo();

  const { data: analysisStatus } = useGetAnalysisStatus(jobId || "", {
    query: {
      enabled: !!jobId,
      queryKey: getGetAnalysisStatusQueryKey(jobId || ""),
      refetchInterval: (query) => {
        const s = query.state.data;
        if (s && (s.job.status === "pending" || s.job.status === "processing")) return 1500;
        return false;
      },
    },
  });

  const isCompleted = analysisStatus?.job?.status === "completed";
  const isAnalyzing =
    analysisStatus?.job?.status === "pending" ||
    analysisStatus?.job?.status === "processing";
  const isFailed = analysisStatus?.job?.status === "failed";

  const { data: rawFrames } = useGetAnalysisFrames(jobId || "", {
    query: {
      enabled: !!jobId && isCompleted,
      queryKey: getGetAnalysisFramesQueryKey(jobId || ""),
    },
  });
  const frames = (rawFrames as DetectionFrame[] | undefined) ?? [];

  const { data: speciesInfo, isLoading: isSpeciesInfoLoading } = useGetSpeciesInfo(
    selectedSpecies || "",
    {
      query: {
        enabled: !!selectedSpecies,
        queryKey: getGetSpeciesInfoQueryKey(selectedSpecies || ""),
      },
    },
  );

  const handleFileDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("video/")) { setFile(f); setAutoPlay(false); }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) { setFile(f); setAutoPlay(false); }
  };

  const handleStartAnalysis = async () => {
    if (!file) return;
    try {
      setAutoPlay(true); // play video immediately
      const res = await uploadMutation.mutateAsync({
        data: { video: file, sampleInterval: sampleInterval[0] } as any,
      });
      setJobId(res.id);
      setSelectedSpecies(null);
    } catch (err) {
      console.error("Upload failed", err);
      setAutoPlay(false);
    }
  };

  const resetAll = () => {
    setFile(null);
    setJobId(null);
    setSelectedSpecies(null);
    setAutoPlay(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // ── Layout: no file selected ────────────────────────────────────────────────
  if (!file) {
    return (
      <div className="space-y-6 animate-in fade-in duration-500">
        <div>
          <h1 className="text-3xl font-serif font-bold">Video Analysis</h1>
          <p className="text-muted-foreground mt-1">
            Upload bird footage for AI species detection and tracking.
          </p>
        </div>

        <Card className="max-w-2xl mx-auto">
          <CardHeader>
            <CardTitle>New Analysis</CardTitle>
            <CardDescription>Drag and drop a video file (MP4, AVI, MOV) to begin.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div
              className="border-2 border-dashed rounded-xl p-14 flex flex-col items-center justify-center text-center cursor-pointer transition-colors border-border hover:border-primary/50 hover:bg-muted/50"
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleFileDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                type="file"
                ref={fileInputRef}
                className="hidden"
                accept="video/mp4,video/avi,video/quicktime"
                onChange={handleFileSelect}
              />
              <div className="h-14 w-14 rounded-full bg-muted flex items-center justify-center text-muted-foreground mb-4">
                <UploadCloud className="h-7 w-7" />
              </div>
              <h3 className="text-lg font-medium">Click or drag a video here</h3>
              <p className="text-sm text-muted-foreground mt-1">MP4 · AVI · MOV — max 500 MB</p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // ── Layout: file selected (all phases with file present) ────────────────────
  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h1 className="text-3xl font-serif font-bold">Video Analysis</h1>
          <p className="text-muted-foreground mt-1">
            Upload bird footage for AI species detection and tracking.
          </p>
        </div>
        {!isAnalyzing && (
          <Button variant="outline" size="sm" onClick={resetAll} className="gap-1.5">
            <UploadCloud className="h-4 w-4" /> Analyse another video
          </Button>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* ── LEFT PANEL ─────────────────────────────────────────────────── */}
        <div className="lg:col-span-1 space-y-4">

          {/* File info + controls (only before job starts) */}
          {!jobId && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <PlaySquare className="h-4 w-4 text-primary" /> Selected File
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-5">
                <div className="bg-muted/40 rounded-lg p-3 flex items-center gap-3">
                  <div className="h-9 w-9 rounded-lg bg-primary/10 flex items-center justify-center text-primary shrink-0">
                    <PlaySquare className="h-5 w-5" />
                  </div>
                  <div className="min-w-0">
                    <p className="text-sm font-medium truncate">{file.name}</p>
                    <p className="text-xs text-muted-foreground">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                  </div>
                </div>

                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium">Sample Interval</label>
                    <span className="text-sm font-mono text-muted-foreground">{sampleInterval[0]}s</span>
                  </div>
                  <Slider
                    value={sampleInterval}
                    onValueChange={setSampleInterval}
                    min={0.5}
                    max={5}
                    step={0.5}
                  />
                  <p className="text-xs text-muted-foreground">
                    How often frames are sampled. Lower = more frames analysed.
                  </p>
                </div>

                <Button
                  className="w-full gap-2"
                  size="lg"
                  disabled={uploadMutation.isPending}
                  onClick={handleStartAnalysis}
                >
                  {uploadMutation.isPending ? (
                    <><Loader2 className="h-4 w-4 animate-spin" /> Uploading…</>
                  ) : (
                    <><Play className="h-4 w-4" /> Start Analysis</>
                  )}
                </Button>
              </CardContent>
            </Card>
          )}

          {/* Progress card during analysis */}
          {isAnalyzing && (
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base">Analysing…</CardTitle>
                  <Badge variant="secondary" className="animate-pulse capitalize">
                    {analysisStatus?.job?.status}
                  </Badge>
                </div>
                <CardDescription className="text-xs mt-0.5">
                  {analysisStatus?.job?.filename || file.name}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">
                      {analysisStatus?.job?.status === "processing"
                        ? `Frame ${analysisStatus.job.processedFrames ?? 0} / ${analysisStatus.job.totalFrames ?? "…"}`
                        : "Queued…"}
                    </span>
                    <span className="font-mono font-medium">
                      {Math.round(analysisStatus?.job?.progress ?? 0)}%
                    </span>
                  </div>
                  <Progress value={analysisStatus?.job?.progress ?? 0} className="h-2" />
                </div>
                <div className="space-y-2 text-xs text-muted-foreground">
                  {[
                    { done: (analysisStatus?.job?.progress ?? 0) >= 15, label: "Frame extraction" },
                    { done: (analysisStatus?.job?.progress ?? 0) >= 80, label: "Gemini 1.5 Flash detection" },
                    { done: (analysisStatus?.job?.progress ?? 0) >= 90, label: "ByteTracker ID assignment" },
                    { done: (analysisStatus?.job?.progress ?? 0) >= 100, label: "Species validation" },
                  ].map(({ done, label }) => (
                    <div key={label} className="flex items-center gap-2">
                      <div className={cn(
                        "h-2 w-2 rounded-full shrink-0",
                        done ? "bg-green-500" : "bg-muted-foreground/30"
                      )} />
                      <span className={done ? "text-foreground" : ""}>{label}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Summary stats after completion */}
          {isCompleted && (
            <div className="space-y-3">
              {[
                { label: "Unique Birds", value: analysisStatus?.summary?.totalBirdsDetected ?? 0 },
                { label: "Species Detected", value: analysisStatus?.summary?.uniqueSpecies ?? 0 },
                { label: "Frames Analysed", value: analysisStatus?.job?.totalFrames ?? 0 },
                {
                  label: "Processing Time",
                  value: `${Math.round(analysisStatus?.summary?.processingDurationSeconds ?? 0)}s`,
                },
              ].map((stat) => (
                <div
                  key={stat.label}
                  className="flex items-center justify-between px-4 py-3 rounded-lg bg-muted/30 border border-border/50"
                >
                  <span className="text-sm text-muted-foreground">{stat.label}</span>
                  <span className="text-lg font-bold font-mono">{stat.value}</span>
                </div>
              ))}
            </div>
          )}

          {/* Species info panel (right-panel on desktop becomes bottom on mobile) */}
          {isCompleted && (
            <div>
              {selectedSpecies ? (
                <Card className="border-primary/20 shadow-md overflow-hidden animate-in fade-in slide-in-from-bottom-4">
                  {isSpeciesInfoLoading ? (
                    <div className="p-5 space-y-4">
                      <Skeleton className="h-7 w-3/4" />
                      <Skeleton className="h-4 w-1/2" />
                      <Skeleton className="h-20 w-full" />
                      <p className="text-xs text-center text-muted-foreground animate-pulse">
                        Generating AI field guide…
                      </p>
                    </div>
                  ) : speciesInfo ? (
                    <>
                      <CardHeader className="bg-primary/5 border-b pb-4">
                        <div className="flex items-center justify-between mb-1">
                          <Badge variant="outline" className="bg-background text-xs">
                            {speciesInfo.conservationStatus}
                          </Badge>
                          <Badge variant="secondary" className="text-xs">
                            {speciesInfo.source === "ai" ? "AI Generated" : "Cached"}
                          </Badge>
                        </div>
                        <div className="flex items-center gap-2 mt-1">
                          <div
                            className="w-3 h-3 rounded-full shrink-0"
                            style={{ backgroundColor: getSpeciesColor(speciesInfo.speciesName) }}
                          />
                          <CardTitle className="text-xl font-serif">{speciesInfo.speciesName}</CardTitle>
                        </div>
                        <CardDescription className="italic text-sm mt-0.5">
                          {speciesInfo.scientificName}
                        </CardDescription>
                      </CardHeader>
                      <div className="p-5 space-y-4 overflow-y-auto max-h-[55vh]">
                        <div>
                          <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1.5 flex items-center gap-1">
                            <Info className="h-3.5 w-3.5" /> Description
                          </h4>
                          <p className="text-sm leading-relaxed">{speciesInfo.description}</p>
                        </div>
                        <div className="grid grid-cols-2 gap-3">
                          <div className="bg-muted/50 p-3 rounded-lg border border-border/50">
                            <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">
                              Habitat
                            </h4>
                            <p className="text-xs leading-relaxed">{speciesInfo.habitat}</p>
                          </div>
                          <div className="bg-muted/50 p-3 rounded-lg border border-border/50">
                            <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">
                              Diet
                            </h4>
                            <p className="text-xs leading-relaxed">{speciesInfo.diet}</p>
                          </div>
                        </div>
                        {speciesInfo.ebirdOccurrences && (
                          <div className="bg-accent/10 p-3 rounded-lg border border-accent/20">
                            <h4 className="text-xs font-semibold uppercase tracking-wider mb-1 text-accent-foreground/70">
                              eBird — North Africa
                            </h4>
                            <p className="text-xs text-accent-foreground">{speciesInfo.ebirdOccurrences}</p>
                          </div>
                        )}
                        <div className="bg-secondary/10 p-4 rounded-lg border border-secondary/20 relative overflow-hidden">
                          <div className="absolute top-0 left-0 w-1 h-full bg-secondary" />
                          <h4 className="text-xs font-semibold text-secondary-foreground mb-1">Fun Fact</h4>
                          <p className="text-xs text-secondary-foreground/90 italic leading-relaxed">
                            {speciesInfo.funFact}
                          </p>
                        </div>
                      </div>
                    </>
                  ) : (
                    <div className="p-6 text-center text-muted-foreground text-sm">
                      Could not load species info.
                    </div>
                  )}
                </Card>
              ) : (
                <Card className="h-40 flex flex-col items-center justify-center text-center p-6 border-dashed border-2 bg-muted/20">
                  <Bird className="h-8 w-8 text-muted mb-2" strokeWidth={1} />
                  <p className="text-sm text-muted-foreground">
                    Click a species badge or table row to open the AI field guide
                  </p>
                </Card>
              )}
            </div>
          )}
        </div>

        {/* ── RIGHT PANEL (video + results table) ───────────────────────── */}
        <div className="lg:col-span-2 space-y-6">

          {/* Video preview — before analysis starts */}
          {!jobId && (
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-base">Video Preview</CardTitle>
                    <CardDescription className="text-xs mt-0.5">
                      Press Start Analysis to begin AI bird detection
                    </CardDescription>
                  </div>
                  <ChevronRight className="h-4 w-4 text-muted-foreground" />
                </div>
              </CardHeader>
              <CardContent>
                <VideoPreview file={file} autoPlay={false} />
              </CardContent>
            </Card>
          )}

          {/* Video preview — while analysis runs (auto-plays) */}
          {isAnalyzing && (
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-base">Video Playback</CardTitle>
                    <CardDescription className="text-xs mt-0.5">
                      Bounding boxes will appear once analysis completes
                    </CardDescription>
                  </div>
                  <Badge variant="secondary" className="text-xs animate-pulse">
                    <Loader2 className="h-3 w-3 mr-1 animate-spin" /> Analysing
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <VideoPreview file={file} autoPlay={autoPlay} />
              </CardContent>
            </Card>
          )}

          {/* Full tracking player — after analysis completes */}
          {isCompleted && (
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-base">Live Playback</CardTitle>
                    <CardDescription className="text-xs mt-0.5">
                      Bounding boxes update frame-by-frame as the video plays
                    </CardDescription>
                  </div>
                  <Badge className="bg-green-600 text-white">
                    <span className="w-1.5 h-1.5 rounded-full bg-white mr-1.5 animate-pulse inline-block" />
                    Real-time tracking
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <VideoPlayer
                  file={file}
                  frames={frames}
                  onSpeciesClick={setSelectedSpecies}
                  selectedSpecies={selectedSpecies}
                />
              </CardContent>
            </Card>
          )}

          {/* ── Results table ───────────────────────────────────────────── */}
          {isCompleted && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <Bird className="h-4 w-4 text-primary" />
                  Detection Results
                </CardTitle>
                <CardDescription>
                  Click any row to open the AI field guide for that species.
                </CardDescription>
              </CardHeader>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Species</TableHead>
                    <TableHead className="text-center">Unique Birds</TableHead>
                    <TableHead className="text-right">Avg Confidence</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {(analysisStatus?.detections ?? []).length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={3} className="text-center text-muted-foreground py-10">
                        <Bird className="h-8 w-8 text-muted mx-auto mb-2" strokeWidth={1} />
                        No birds detected in this footage.
                      </TableCell>
                    </TableRow>
                  ) : (
                    (analysisStatus?.detections ?? []).map((d) => (
                      <TableRow
                        key={d.species}
                        className={cn(
                          "cursor-pointer hover:bg-muted/50 transition-colors",
                          selectedSpecies === d.species && "bg-primary/5",
                        )}
                        onClick={() => setSelectedSpecies(d.species)}
                      >
                        <TableCell>
                          <div className="flex items-center gap-2.5">
                            <div
                              className="w-3 h-3 rounded-full shrink-0"
                              style={{
                                backgroundColor: d.color || getSpeciesColor(d.species),
                              }}
                            />
                            <span className="font-medium">{d.species}</span>
                          </div>
                        </TableCell>
                        <TableCell className="text-center">
                          <Badge variant="secondary" className="font-mono tabular-nums">
                            {d.totalCount}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex items-center justify-end gap-2">
                            <span className="text-sm text-muted-foreground tabular-nums">
                              {Math.round(d.averageConfidence * 100)}%
                            </span>
                            <Progress value={d.averageConfidence * 100} className="w-16 h-1.5" />
                          </div>
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </Card>
          )}
        </div>
      </div>

      {/* Failed state */}
      {isFailed && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Analysis Failed</AlertTitle>
          <AlertDescription>
            There was an error processing this video. Please try again with a different file.
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}
