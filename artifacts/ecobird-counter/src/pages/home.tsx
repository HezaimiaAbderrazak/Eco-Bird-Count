import { useState, useRef, useEffect } from "react";
import { UploadCloud, FileVideo, Activity, Info, AlertCircle, PlaySquare, Bird } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useUploadVideo, useGetAnalysisStatus, getGetAnalysisStatusQueryKey, useGetSpeciesInfo, getGetSpeciesInfoQueryKey } from "@workspace/api-client-react";
import { getSpeciesColor } from "@/lib/colors";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [sampleInterval, setSampleInterval] = useState([2.0]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [selectedSpecies, setSelectedSpecies] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const uploadMutation = useUploadVideo();
  
  const { data: analysisStatus, isError: analysisError } = useGetAnalysisStatus(jobId || "", {
    query: {
      enabled: !!jobId,
      queryKey: getGetAnalysisStatusQueryKey(jobId || ""),
      refetchInterval: (query) => {
        const state = query.state.data;
        if (state && (state.job.status === "pending" || state.job.status === "processing")) {
          return 2000;
        }
        return false;
      }
    }
  });

  const { data: speciesInfo, isLoading: isSpeciesInfoLoading } = useGetSpeciesInfo(selectedSpecies || "", {
    query: {
      enabled: !!selectedSpecies,
      queryKey: getGetSpeciesInfoQueryKey(selectedSpecies || "")
    }
  });

  const handleFileDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith("video/")) {
      setFile(droppedFile);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    
    try {
      // Construct FormData and let mutateAsync handle it correctly based on the hook's signature
      const res = await uploadMutation.mutateAsync({ data: { video: file, sampleInterval: sampleInterval[0] } as any });
      setJobId(res.id);
      setSelectedSpecies(null);
    } catch (error) {
      console.error("Upload failed", error);
    }
  };

  const isAnalyzing = analysisStatus?.job?.status === "pending" || analysisStatus?.job?.status === "processing";
  const isCompleted = analysisStatus?.job?.status === "completed";
  const isFailed = analysisStatus?.job?.status === "failed";

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div>
        <h1 className="text-3xl font-serif font-bold text-foreground">Video Analysis</h1>
        <p className="text-muted-foreground mt-2">Upload footage for AI-powered species detection and identification.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-8">
          {/* Upload Card */}
          <Card>
            <CardHeader>
              <CardTitle>New Analysis</CardTitle>
              <CardDescription>Drag and drop a video file (MP4, AVI, MOV) to begin.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {!isAnalyzing && !isCompleted && !isFailed ? (
                <>
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
                        <p className="text-sm text-muted-foreground mt-1">
                          {(file.size / (1024 * 1024)).toFixed(2)} MB
                        </p>
                      </>
                    ) : (
                      <>
                        <div className="h-12 w-12 rounded-full bg-muted flex items-center justify-center text-muted-foreground mb-4">
                          <UploadCloud className="h-6 w-6" />
                        </div>
                        <h3 className="text-lg font-medium">Click or drag video here</h3>
                        <p className="text-sm text-muted-foreground mt-1">Maximum file size 500MB</p>
                      </>
                    )}
                  </div>

                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <label className="text-sm font-medium">Sample Interval</label>
                      <span className="text-sm text-muted-foreground">{sampleInterval[0]}s</span>
                    </div>
                    <Slider
                      value={sampleInterval}
                      onValueChange={setSampleInterval}
                      min={0.5}
                      max={5.0}
                      step={0.5}
                    />
                    <p className="text-xs text-muted-foreground">
                      How frequently to extract frames for analysis. Lower intervals provide better accuracy but take longer to process.
                    </p>
                  </div>
                  
                  <Button 
                    className="w-full" 
                    size="lg" 
                    disabled={!file || uploadMutation.isPending}
                    onClick={handleUpload}
                  >
                    {uploadMutation.isPending ? "Uploading..." : "Start Analysis"}
                  </Button>
                </>
              ) : (
                <div className="py-6 space-y-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-medium">{analysisStatus?.job?.filename || file?.name || "Video File"}</h3>
                      <p className="text-sm text-muted-foreground">
                        {analysisStatus?.job?.status === 'processing' ? 'Processing frames...' : 
                         analysisStatus?.job?.status === 'pending' ? 'Queued for processing...' : 
                         analysisStatus?.job?.status === 'completed' ? 'Analysis complete' : 'Analysis failed'}
                      </p>
                    </div>
                    <Badge variant={isCompleted ? "default" : isFailed ? "destructive" : "secondary"} className="capitalize">
                      {analysisStatus?.job?.status || "uploading"}
                    </Badge>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Progress</span>
                      <span className="font-medium">{Math.round(analysisStatus?.job?.progress || 0)}%</span>
                    </div>
                    <Progress value={analysisStatus?.job?.progress || 0} className="h-2" />
                  </div>

                  {isCompleted && (
                    <Button variant="outline" className="w-full" onClick={() => {
                      setJobId(null);
                      setFile(null);
                      setSelectedSpecies(null);
                    }}>
                      Analyze Another Video
                    </Button>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Results Section */}
          {isCompleted && analysisStatus?.detections && (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <Card>
                  <CardContent className="p-4">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Total Birds</p>
                    <p className="text-2xl font-bold mt-1">{analysisStatus.summary?.totalBirdsDetected || 0}</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Unique Species</p>
                    <p className="text-2xl font-bold mt-1">{analysisStatus.summary?.uniqueSpecies || 0}</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Most Common</p>
                    <p className="text-lg font-bold mt-1 truncate">{analysisStatus.summary?.mostCommonSpecies || "-"}</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-4">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Duration</p>
                    <p className="text-2xl font-bold mt-1">{analysisStatus.summary?.processingDurationSeconds || 0}s</p>
                  </CardContent>
                </Card>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle>Species Detected</CardTitle>
                  <CardDescription>Click a row to view detailed species information.</CardDescription>
                </CardHeader>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Species</TableHead>
                      <TableHead className="text-right">Count</TableHead>
                      <TableHead className="text-right">Confidence</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {analysisStatus.detections.length === 0 && (
                      <TableRow>
                        <TableCell colSpan={3} className="text-center text-muted-foreground py-8">
                          No birds detected in this video.
                        </TableCell>
                      </TableRow>
                    )}
                    {analysisStatus.detections.map((d) => (
                      <TableRow 
                        key={d.species} 
                        className={cn(
                          "cursor-pointer hover:bg-muted/50 transition-colors",
                          selectedSpecies === d.species && "bg-muted/80"
                        )}
                        onClick={() => setSelectedSpecies(d.species)}
                      >
                        <TableCell className="font-medium flex items-center gap-3">
                          <div 
                            className="w-3 h-3 rounded-full" 
                            style={{ backgroundColor: d.color || getSpeciesColor(d.species) }} 
                          />
                          {d.species}
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
              </Card>
            </div>
          )}

          {isFailed && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Analysis Failed</AlertTitle>
              <AlertDescription>
                There was an error processing the video. Please try again with a different file.
              </AlertDescription>
            </Alert>
          )}
        </div>

        {/* Species Info Panel */}
        <div className="lg:col-span-1">
          {selectedSpecies ? (
            <Card className="sticky top-6 h-[calc(100vh-8rem)] flex flex-col overflow-hidden animate-in fade-in slide-in-from-right-8 border-primary/20 shadow-md">
              {isSpeciesInfoLoading ? (
                <div className="p-6 space-y-6">
                  <div className="space-y-3">
                    <Skeleton className="h-8 w-3/4" />
                    <Skeleton className="h-4 w-1/2" />
                  </div>
                  <Skeleton className="h-[200px] w-full rounded-lg" />
                  <div className="space-y-4">
                    <Skeleton className="h-4 w-full" />
                    <Skeleton className="h-4 w-full" />
                    <Skeleton className="h-4 w-4/5" />
                  </div>
                </div>
              ) : speciesInfo ? (
                <>
                  <CardHeader className="bg-primary/5 border-b pb-4">
                    <div className="flex items-center justify-between mb-1">
                      <Badge variant="outline" className="bg-background">
                        {speciesInfo.conservationStatus}
                      </Badge>
                      <Badge variant="secondary" className="text-xs">
                        {speciesInfo.source === "ai" ? "AI Generated" : "Cached"}
                      </Badge>
                    </div>
                    <CardTitle className="text-2xl font-serif">{speciesInfo.speciesName}</CardTitle>
                    <CardDescription className="italic text-muted-foreground text-sm">
                      {speciesInfo.scientificName}
                    </CardDescription>
                  </CardHeader>
                  <div className="flex-1 overflow-y-auto p-6 space-y-6">
                    <div>
                      <h4 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground mb-2 flex items-center gap-2">
                        <Info className="h-4 w-4" /> Description
                      </h4>
                      <p className="text-sm leading-relaxed">{speciesInfo.description}</p>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-muted/50 p-3 rounded-lg border border-border/50">
                        <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">Habitat</h4>
                        <p className="text-sm">{speciesInfo.habitat}</p>
                      </div>
                      <div className="bg-muted/50 p-3 rounded-lg border border-border/50">
                        <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">Diet</h4>
                        <p className="text-sm">{speciesInfo.diet}</p>
                      </div>
                    </div>

                    {speciesInfo.ebirdOccurrences && (
                      <div className="bg-accent/10 text-accent-foreground p-3 rounded-lg border border-accent/20">
                        <h4 className="text-xs font-semibold uppercase tracking-wider mb-1 opacity-80">eBird Data</h4>
                        <p className="text-sm font-medium">{speciesInfo.ebirdOccurrences}</p>
                      </div>
                    )}

                    <div className="bg-secondary/10 p-4 rounded-lg border border-secondary/20 relative overflow-hidden">
                      <div className="absolute top-0 left-0 w-1 h-full bg-secondary"></div>
                      <h4 className="text-sm font-semibold text-secondary-foreground mb-1">Fun Fact</h4>
                      <p className="text-sm text-secondary-foreground/90 italic">{speciesInfo.funFact}</p>
                    </div>
                  </div>
                </>
              ) : (
                <div className="flex-1 flex items-center justify-center p-6 text-center text-muted-foreground">
                  <p>Could not load species information.</p>
                </div>
              )}
            </Card>
          ) : (
            <Card className="sticky top-6 h-[500px] flex flex-col items-center justify-center text-center p-8 border-dashed border-2 bg-muted/20">
              <Bird className="h-16 w-16 text-muted mb-4" strokeWidth={1} />
              <h3 className="font-medium text-lg text-muted-foreground">Field Guide</h3>
              <p className="text-sm text-muted-foreground/70 mt-2">
                Select a species from the results table to view its detailed profile, including habitat, diet, and conservation status.
              </p>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
