import { format } from "date-fns";
import { History, FileVideo, Clock, CheckCircle2, AlertCircle, Loader2 } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { useGetRecentJobs } from "@workspace/api-client-react";
import { Link } from "wouter";

export default function HistoryPage() {
  const { data: jobs, isLoading } = useGetRecentJobs();

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div>
        <h1 className="text-3xl font-serif font-bold text-foreground flex items-center gap-3">
          <History className="h-8 w-8 text-primary" />
          Analysis History
        </h1>
        <p className="text-muted-foreground mt-2">View records of your past video analyses.</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Recent Jobs</CardTitle>
          <CardDescription>A log of your recent AI detection sessions.</CardDescription>
        </CardHeader>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>File</TableHead>
              <TableHead>Date</TableHead>
              <TableHead>Status</TableHead>
              <TableHead className="text-right">Processed Frames</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading ? (
              Array.from({ length: 5 }).map((_, i) => (
                <TableRow key={i}>
                  <TableCell><Skeleton className="h-5 w-48" /></TableCell>
                  <TableCell><Skeleton className="h-5 w-32" /></TableCell>
                  <TableCell><Skeleton className="h-6 w-24 rounded-full" /></TableCell>
                  <TableCell className="text-right"><Skeleton className="h-5 w-16 ml-auto" /></TableCell>
                </TableRow>
              ))
            ) : Array.isArray(jobs) && jobs.length > 0 ? (
              jobs.map((job) => (
                <TableRow key={job.id}>
                  <TableCell className="font-medium flex items-center gap-3">
                    <FileVideo className="h-4 w-4 text-muted-foreground" />
                    {job.filename}
                  </TableCell>
                  <TableCell className="text-muted-foreground">
                    {format(new Date(job.createdAt), "MMM d, yyyy 'at' h:mm a")}
                  </TableCell>
                  <TableCell>
                    {job.status === "completed" && (
                      <Badge variant="default" className="bg-primary/10 text-primary border-primary/20 hover:bg-primary/20">
                        <CheckCircle2 className="mr-1 h-3 w-3" /> Completed
                      </Badge>
                    )}
                    {job.status === "failed" && (
                      <Badge variant="destructive" className="bg-destructive/10 text-destructive border-destructive/20 hover:bg-destructive/20">
                        <AlertCircle className="mr-1 h-3 w-3" /> Failed
                      </Badge>
                    )}
                    {(job.status === "processing" || job.status === "pending") && (
                      <div className="flex items-center gap-2 max-w-[120px]">
                        <Badge variant="secondary" className="bg-secondary/10 text-secondary-foreground border-secondary/20">
                          <Loader2 className="mr-1 h-3 w-3 animate-spin" /> In Progress
                        </Badge>
                      </div>
                    )}
                  </TableCell>
                  <TableCell className="text-right text-muted-foreground">
                    {job.processedFrames !== null && job.totalFrames !== null ? (
                      <span className="font-mono text-sm">{job.processedFrames} / {job.totalFrames}</span>
                    ) : (
                      "-"
                    )}
                  </TableCell>
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={4} className="h-32 text-center text-muted-foreground">
                  <div className="flex flex-col items-center justify-center gap-2">
                    <FileVideo className="h-8 w-8 text-muted" />
                    <p>No analysis jobs found.</p>
                    <Link href="/" className="text-primary hover:underline text-sm font-medium mt-1">Start a new analysis</Link>
                  </div>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </Card>
    </div>
  );
}
