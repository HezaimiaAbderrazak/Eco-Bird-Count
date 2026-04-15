import { useState, useRef, useEffect, useCallback } from "react";
import { UploadCloud, PlaySquare, Bird, AlertCircle, Info, Play, Pause, RotateCcw } from "lucide-react";
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
  const containerRef = useRef<HTMLDivElement>(null);
  const videoUrl = useRef<string>("");
  const animFrameRef = useRef<number>(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [liveCounters, setLiveCounters] = useState<Record<string, { count: number; color: string }>>({});
  const [currentDetections, setCurrentDetections] = useState<BirdDetection[]>([]);

  useEffect(() => {
    videoUrl.current = URL.createObjectURL(file);
    return () => URL.revokeObjectURL(videoUrl.current);
  }, [file]);

  const getCurrentFrame = useCallback(
    (time: number): DetectionFrame | null => {
      if (!frames.length) return null;
      let best: DetectionFrame | null = null;
      let minDiff = Infinity;
      for (const f of frames) {
        const diff = Math.abs(f.timestamp - time);
        if (diff < minDiff) {
          minDiff = diff;
          best = f;
        }
      }
      return best;
    },
    [frames]
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
    if (!frame || !frame.detections.length) return;

    setCurrentDetections(frame.detections);

    // Update live counters for all frames up to current time
    const seenTrackIds = new Set<string>();
    const counters: Record<string, { count: number; color: string }> = {};
    for (const f of frames) {
      if (f.timestamp > t + 0.5) break;
      for (const d of f.detections) {
        const key = `${d.species}-${d.trackId}`;
        if (!seenTrackIds.has(key)) {
          seenTrackIds.add(key);
          if (!counters[d.species]) counters[d.species] = { count: 0, color: d.color };
          counters[d.species].count++;
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

      // Glow effect
      ctx.shadowColor = color;
      ctx.shadowBlur = 12;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(px, py, pw, ph);
      ctx.shadowBlur = 0;

      // Corner marks
      const cLen = Math.min(pw, ph) * 0.25;
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      // TL
      ctx.beginPath(); ctx.moveTo(px, py + cLen); ctx.lineTo(px, py); ctx.lineTo(px + cLen, py); ctx.stroke();
      // TR
      ctx.beginPath(); ctx.moveTo(px + pw - cLen, py); ctx.lineTo(px + pw, py); ctx.lineTo(px + pw, py + cLen); ctx.stroke();
      // BL
      ctx.beginPath(); ctx.moveTo(px, py + ph - cLen); ctx.lineTo(px, py + ph); ctx.lineTo(px + cLen, py + ph); ctx.stroke();
      // BR
      ctx.beginPath(); ctx.moveTo(px + pw - cLen, py + ph); ctx.lineTo(px + pw, py + ph); ctx.lineTo(px + pw, py + ph - cLen); ctx.stroke();

      // Label
      const label = `${det.species} ${Math.round(det.confidence * 100)}%`;
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

      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, lx + 5, ly + th - 5);

      // Track ID dot
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(px + pw - 6, py + 6, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#ffffff";
      ctx.font = `bold ${Math.max(8, fontSize - 3)}px monospace`;
      ctx.textAlign = "center";
      ctx.fillText(`${det.trackId}`, px + pw - 6, py + 10);
      ctx.textAlign = "left";
    }
  }, [frames, getCurrentFrame]);

  useEffect(() => {
    let raf: number;
    const loop = () => {
      drawFrame();
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    animFrameRef.current = raf;
    return () => cancelAnimationFrame(raf);
  }, [drawFrame]);

  const handleTimeUpdate = () => {
    if (videoRef.current) setCurrentTime(videoRef.current.currentTime);
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) setDuration(videoRef.current.duration);
  };

  const togglePlay = () => {
    if (!videoRef.current) return;
    if (videoRef.current.paused) {
      videoRef.current.play();
      setIsPlaying(true);
    } else {
      videoRef.current.pause();
      setIsPlaying(false);
    }
  };

  const handleSeek = (val: number[]) => {
    if (videoRef.current) {
      videoRef.current.currentTime = val[0];
      setCurrentTime(val[0]);
    }
  };

  const handleRestart = () => {
    if (videoRef.current) {
      videoRef.current.currentTime = 0;
      videoRef.current.play();
      setIsPlaying(true);
    }
  };

  const handleEnded = () => setIsPlaying(false);

  const fmt = (s: number) =>
    `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

  return (
    <div className="space-y-4">
      {/* Video + Canvas overlay */}
      <div
        ref={containerRef}
        className="relative rounded-xl overflow-hidden bg-black shadow-lg border border-border"
        style={{ aspectRatio: "16/9" }}
      >
        <video
          ref={videoRef}
          src={videoUrl.current}
          className="w-full h-full object-contain"
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onEnded={handleEnded}
          playsInline
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
          style={{ zIndex: 10 }}
        />

        {/* Live detection badge overlay */}
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

        {/* Frame timestamp */}
        <div className="absolute bottom-3 right-3 z-20 bg-black/60 text-white text-xs font-mono px-2 py-1 rounded">
          {fmt(currentTime)} / {fmt(duration || 0)}
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3">
        <Button size="sm" variant="outline" onClick={togglePlay} className="gap-1.5 shrink-0">
          {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          {isPlaying ? "Pause" : "Play"}
        </Button>
        <Button size="sm" variant="ghost" onClick={handleRestart} className="shrink-0 px-2">
          <RotateCcw className="h-4 w-4" />
        </Button>
        <Slider
          value={[currentTime]}
          onValueChange={handleSeek}
          min={0}
          max={duration || 100}
          step={0.1}
          className="flex-1"
        />
      </div>

      {/* Live Species Counters */}
      {Object.keys(liveCounters).length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
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
                    : "border-border bg-card hover:border-primary/40"
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

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [sampleInterval, setSampleInterval] = useState([1.5]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [selectedSpecies, setSelectedSpecies] = useState<string | null>(null);
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
  const isAnalyzing = analysisStatus?.job?.status === "pending" || analysisStatus?.job?.status === "processing";
  const isFailed = analysisStatus?.job?.status === "failed";

  const { data: rawFrames } = useGetAnalysisFrames(jobId || "", {
    query: {
      enabled: !!jobId && isCompleted,
      queryKey: getGetAnalysisFramesQueryKey(jobId || ""),
    },
  });

  const frames = (rawFrames as DetectionFrame[] | undefined) ?? [];

  const { data: speciesInfo, isLoading: isSpeciesInfoLoading } = useGetSpeciesInfo(selectedSpecies || "", {
    query: {
      enabled: !!selectedSpecies,
      queryKey: getGetSpeciesInfoQueryKey(selectedSpecies || ""),
    },
  });

  const handleFileDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("video/")) setFile(f);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) setFile(f);
  };

  const handleUpload = async () => {
    if (!file) return;
    try {
      const res = await uploadMutation.mutateAsync({ data: { video: file, sampleInterval: sampleInterval[0] } as any });
      setJobId(res.id);
      setSelectedSpecies(null);
    } catch (err) {
      console.error("Upload failed", err);
    }
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div>
        <h1 className="text-3xl font-serif font-bold">Video Analysis</h1>
        <p className="text-muted-foreground mt-1">Upload footage for real-time AI species detection and identification.</p>
      </div>

      {/* Upload / Progress */}
      {!jobId ? (
        <Card>
          <CardHeader>
            <CardTitle>New Analysis</CardTitle>
            <CardDescription>Drag and drop a video file (MP4, AVI, MOV) to begin.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div
              className={cn(
                "border-2 border-dashed rounded-xl p-10 flex flex-col items-center justify-center text-center cursor-pointer transition-colors",
                file ? "border-primary bg-primary/5" : "border-border hover:border-primary/50 hover:bg-muted/50"
              )}
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
              {file ? (
                <>
                  <div className="h-12 w-12 rounded-full bg-primary/20 flex items-center justify-center text-primary mb-4">
                    <PlaySquare className="h-6 w-6" />
                  </div>
                  <h3 className="text-lg font-medium">{file.name}</h3>
                  <p className="text-sm text-muted-foreground mt-1">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                </>
              ) : (
                <>
                  <div className="h-12 w-12 rounded-full bg-muted flex items-center justify-center text-muted-foreground mb-4">
                    <UploadCloud className="h-6 w-6" />
                  </div>
                  <h3 className="text-lg font-medium">Click or drag video here</h3>
                  <p className="text-sm text-muted-foreground mt-1">Maximum file size 500 MB</p>
                </>
              )}
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Sample Interval</label>
                <span className="text-sm text-muted-foreground">{sampleInterval[0]}s</span>
              </div>
              <Slider value={sampleInterval} onValueChange={setSampleInterval} min={0.5} max={5} step={0.5} />
              <p className="text-xs text-muted-foreground">
                How frequently to sample frames for analysis. Lower = more precision.
              </p>
            </div>

            <Button className="w-full" size="lg" disabled={!file || uploadMutation.isPending} onClick={handleUpload}>
              {uploadMutation.isPending ? "Uploading..." : "Start Analysis"}
            </Button>
          </CardContent>
        </Card>
      ) : isAnalyzing ? (
        <Card>
          <CardContent className="py-8 space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-medium">{analysisStatus?.job?.filename || file?.name}</h3>
                <p className="text-sm text-muted-foreground mt-0.5">
                  {analysisStatus?.job?.status === "processing" ? `Processing frame ${analysisStatus.job.processedFrames || 0} of ${analysisStatus.job.totalFrames || "..."}` : "Queued for processing..."}
                </p>
              </div>
              <Badge variant="secondary" className="capitalize animate-pulse">
                {analysisStatus?.job?.status}
              </Badge>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Progress</span>
                <span className="font-mono font-medium">{Math.round(analysisStatus?.job?.progress || 0)}%</span>
              </div>
              <Progress value={analysisStatus?.job?.progress || 0} className="h-2" />
            </div>
            <p className="text-xs text-muted-foreground text-center">
              The system is simulating YOLOv8 frame sampling + ByteTrack identity tracking...
            </p>
          </CardContent>
        </Card>
      ) : null}

      {/* Main content: video player + info panel */}
      {isCompleted && file && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left: video player + stats table */}
          <div className="lg:col-span-2 space-y-6 animate-in fade-in slide-in-from-bottom-4">
            <Card>
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-base">Live Playback</CardTitle>
                    <CardDescription className="text-xs mt-0.5">
                      Bounding boxes update frame-by-frame as the video plays
                    </CardDescription>
                  </div>
                  <Badge variant="default" className="bg-green-600">
                    <span className="w-1.5 h-1.5 rounded-full bg-white mr-1.5 animate-pulse inline-block" />
                    Real-time
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

            {/* Summary stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {[
                { label: "Total Birds", value: analysisStatus?.summary?.totalBirdsDetected ?? 0 },
                { label: "Species", value: analysisStatus?.summary?.uniqueSpecies ?? 0 },
                { label: "Frames", value: analysisStatus?.job?.totalFrames ?? 0 },
                { label: "Duration", value: `${Math.round(analysisStatus?.summary?.processingDurationSeconds ?? 0)}s` },
              ].map((stat) => (
                <Card key={stat.label}>
                  <CardContent className="p-4">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">{stat.label}</p>
                    <p className="text-2xl font-bold mt-1">{stat.value}</p>
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* Species table */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Detected Species</CardTitle>
                <CardDescription>Click a row to view the AI field guide entry.</CardDescription>
              </CardHeader>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableHead>Species</TableHead>
                    <TableHead className="text-right">Count</TableHead>
                    <TableHead className="text-right">Avg Confidence</TableHead>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {(analysisStatus?.detections ?? []).length === 0 && (
                    <TableRow>
                      <TableCell colSpan={3} className="text-center text-muted-foreground py-8">No birds detected.</TableCell>
                    </TableRow>
                  )}
                  {(analysisStatus?.detections ?? []).map((d) => (
                    <TableRow
                      key={d.species}
                      className={cn(
                        "cursor-pointer hover:bg-muted/50 transition-colors",
                        selectedSpecies === d.species && "bg-muted/80"
                      )}
                      onClick={() => setSelectedSpecies(d.species)}
                    >
                      <TableCell className="font-medium">
                        <div className="flex items-center gap-2.5">
                          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: d.color || getSpeciesColor(d.species) }} />
                          {d.species}
                        </div>
                      </TableCell>
                      <TableCell className="text-right">{d.totalCount}</TableCell>
                      <TableCell className="text-right">
                        <div className="flex items-center justify-end gap-2">
                          <span className="text-sm text-muted-foreground">{Math.round(d.averageConfidence * 100)}%</span>
                          <Progress value={d.averageConfidence * 100} className="w-16 h-1.5" />
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              <div className="p-3 border-t">
                <Button variant="ghost" size="sm" className="w-full text-muted-foreground" onClick={() => { setJobId(null); setFile(null); setSelectedSpecies(null); }}>
                  Analyze another video
                </Button>
              </div>
            </Card>
          </div>

          {/* Right: species info panel */}
          <div className="lg:col-span-1">
            {selectedSpecies ? (
              <Card className="sticky top-6 flex flex-col overflow-hidden animate-in fade-in slide-in-from-right-8 border-primary/20 shadow-md">
                {isSpeciesInfoLoading ? (
                  <div className="p-6 space-y-6">
                    <div className="space-y-3">
                      <Skeleton className="h-8 w-3/4" />
                      <Skeleton className="h-4 w-1/2" />
                    </div>
                    <div className="space-y-3">
                      <Skeleton className="h-4 w-full" />
                      <Skeleton className="h-4 w-full" />
                      <Skeleton className="h-4 w-4/5" />
                    </div>
                    <p className="text-xs text-center text-muted-foreground animate-pulse">Generating AI field guide...</p>
                  </div>
                ) : speciesInfo ? (
                  <>
                    <CardHeader className="bg-primary/5 border-b pb-4">
                      <div className="flex items-center justify-between mb-1">
                        <Badge variant="outline" className="bg-background text-xs">{speciesInfo.conservationStatus}</Badge>
                        <Badge variant="secondary" className="text-xs">{speciesInfo.source === "ai" ? "AI Generated" : "Cached"}</Badge>
                      </div>
                      <div className="flex items-center gap-2 mt-1">
                        <div className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: getSpeciesColor(speciesInfo.speciesName) }} />
                        <CardTitle className="text-xl font-serif">{speciesInfo.speciesName}</CardTitle>
                      </div>
                      <CardDescription className="italic text-sm mt-0.5">{speciesInfo.scientificName}</CardDescription>
                    </CardHeader>
                    <div className="p-5 space-y-5 overflow-y-auto max-h-[60vh]">
                      <div>
                        <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2 flex items-center gap-1.5">
                          <Info className="h-3.5 w-3.5" /> Description
                        </h4>
                        <p className="text-sm leading-relaxed">{speciesInfo.description}</p>
                      </div>

                      <div className="grid grid-cols-2 gap-3">
                        <div className="bg-muted/50 p-3 rounded-lg border border-border/50">
                          <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">Habitat</h4>
                          <p className="text-xs leading-relaxed">{speciesInfo.habitat}</p>
                        </div>
                        <div className="bg-muted/50 p-3 rounded-lg border border-border/50">
                          <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">Diet</h4>
                          <p className="text-xs leading-relaxed">{speciesInfo.diet}</p>
                        </div>
                      </div>

                      {speciesInfo.ebirdOccurrences && (
                        <div className="bg-accent/10 p-3 rounded-lg border border-accent/20">
                          <h4 className="text-xs font-semibold uppercase tracking-wider mb-1 text-accent-foreground/70">eBird — North Africa</h4>
                          <p className="text-xs text-accent-foreground">{speciesInfo.ebirdOccurrences}</p>
                        </div>
                      )}

                      <div className="bg-secondary/10 p-4 rounded-lg border border-secondary/20 relative overflow-hidden">
                        <div className="absolute top-0 left-0 w-1 h-full bg-secondary" />
                        <h4 className="text-xs font-semibold text-secondary-foreground mb-1">Fun Fact</h4>
                        <p className="text-xs text-secondary-foreground/90 italic leading-relaxed">{speciesInfo.funFact}</p>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="p-6 text-center text-muted-foreground text-sm">Could not load species info.</div>
                )}
              </Card>
            ) : (
              <Card className="sticky top-6 h-64 flex flex-col items-center justify-center text-center p-8 border-dashed border-2 bg-muted/20">
                <Bird className="h-12 w-12 text-muted mb-3" strokeWidth={1} />
                <h3 className="font-medium text-muted-foreground">Field Guide</h3>
                <p className="text-xs text-muted-foreground/70 mt-1.5">
                  Click a species badge during playback or a row in the table to open the AI field guide.
                </p>
              </Card>
            )}
          </div>
        </div>
      )}

      {isFailed && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Analysis Failed</AlertTitle>
          <AlertDescription>There was an error processing the video. Please try again.</AlertDescription>
        </Alert>
      )}
    </div>
  );
}
