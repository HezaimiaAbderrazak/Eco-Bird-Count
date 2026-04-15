import { BarChart2, Activity, Bird, Eye } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useGetSpeciesStats } from "@workspace/api-client-react";
import { Skeleton } from "@/components/ui/skeleton";
import { getSpeciesColor } from "@/lib/colors";
import { format, parseISO } from "date-fns";

export default function StatsPage() {
  const { data: stats, isLoading } = useGetSpeciesStats();

  const maxActivity = (stats?.recentActivity ?? []).reduce((max, day) => Math.max(max, day.detections), 0) || 1;
  const maxSpeciesCount = (stats?.topSpecies ?? []).reduce((max, s) => Math.max(max, s.count), 0) || 1;

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div>
        <h1 className="text-3xl font-serif font-bold text-foreground flex items-center gap-3">
          <BarChart2 className="h-8 w-8 text-primary" />
          Global Statistics
        </h1>
        <p className="text-muted-foreground mt-2">Aggregate data from all analysis sessions.</p>
      </div>

      {isLoading ? (
        <div className="space-y-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Skeleton className="h-32 w-full rounded-xl" />
            <Skeleton className="h-32 w-full rounded-xl" />
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <Skeleton className="h-[400px] w-full rounded-xl" />
            <Skeleton className="h-[400px] w-full rounded-xl" />
          </div>
        </div>
      ) : stats ? (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card className="bg-primary/5 border-primary/10">
              <CardContent className="p-6 flex items-center gap-6">
                <div className="h-16 w-16 rounded-full bg-primary/20 flex items-center justify-center text-primary">
                  <Activity className="h-8 w-8" />
                </div>
                <div>
                  <p className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">Total Analyses</p>
                  <p className="text-4xl font-serif font-bold mt-1 text-primary">{(stats.totalAnalyses ?? 0).toLocaleString()}</p>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-secondary/5 border-secondary/10">
              <CardContent className="p-6 flex items-center gap-6">
                <div className="h-16 w-16 rounded-full bg-secondary/20 flex items-center justify-center text-secondary-foreground">
                  <Eye className="h-8 w-8" />
                </div>
                <div>
                  <p className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">Total Detections</p>
                  <p className="text-4xl font-serif font-bold mt-1 text-secondary-foreground">{(stats.totalDetections ?? 0).toLocaleString()}</p>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Bird className="h-5 w-5 text-primary" />
                  Top Species
                </CardTitle>
                <CardDescription>Most frequently detected birds across all footage.</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-6 mt-4">
                  {(stats.topSpecies ?? []).length > 0 ? (stats.topSpecies ?? []).map((species, index) => {
                    const color = getSpeciesColor(species.species);
                    const percentage = Math.max(2, (species.count / maxSpeciesCount) * 100);
                    return (
                      <div key={species.species} className="space-y-2">
                        <div className="flex justify-between items-end text-sm">
                          <span className="font-medium flex items-center gap-2">
                            <span className="text-muted-foreground w-4">{index + 1}.</span> 
                            {species.species}
                          </span>
                          <span className="font-mono text-muted-foreground">{species.count.toLocaleString()}</span>
                        </div>
                        <div className="w-full bg-muted rounded-full h-2.5 overflow-hidden">
                          <div 
                            className="h-full rounded-full transition-all duration-1000 ease-out"
                            style={{ width: `${percentage}%`, backgroundColor: color }}
                          />
                        </div>
                      </div>
                    )
                  }) : (
                    <div className="text-center py-8 text-muted-foreground">No species data available yet.</div>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="h-5 w-5 text-primary" />
                  Recent Activity
                </CardTitle>
                <CardDescription>Detections over the last 7 days.</CardDescription>
              </CardHeader>
              <CardContent className="h-[300px] flex items-end justify-between gap-2 pt-8 pb-2">
                {(stats.recentActivity ?? []).length > 0 ? (stats.recentActivity ?? []).map((day) => {
                  const heightPercentage = Math.max(5, (day.detections / maxActivity) * 100);
                  return (
                    <div key={day.date} className="flex flex-col items-center flex-1 group">
                      <div className="relative w-full flex justify-center h-[200px] items-end">
                        {/* Tooltip on hover */}
                        <div className="absolute -top-10 bg-popover text-popover-foreground border shadow-sm px-2 py-1 rounded text-xs opacity-0 group-hover:opacity-100 transition-opacity z-10 whitespace-nowrap pointer-events-none">
                          {day.detections} detections
                        </div>
                        <div 
                          className="w-full max-w-[40px] bg-primary/20 hover:bg-primary/40 rounded-t-sm transition-all duration-500 ease-out relative overflow-hidden"
                          style={{ height: `${heightPercentage}%` }}
                        >
                          <div className="absolute bottom-0 w-full bg-primary/60" style={{ height: '4px' }}></div>
                        </div>
                      </div>
                      <span className="text-xs text-muted-foreground mt-4 font-medium rotate-[-45deg] origin-top-left translate-x-2">
                        {format(parseISO(day.date), "MMM d")}
                      </span>
                    </div>
                  )
                }) : (
                  <div className="w-full text-center py-8 text-muted-foreground self-center">No activity data available yet.</div>
                )}
              </CardContent>
            </Card>
          </div>
        </>
      ) : (
        <div className="text-center py-12 text-muted-foreground">Failed to load statistics.</div>
      )}
    </div>
  );
}
