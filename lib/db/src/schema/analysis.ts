import { pgTable, text, serial, real, integer, timestamp, jsonb } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";

export const analysisJobsTable = pgTable("analysis_jobs", {
  id: text("id").primaryKey(),
  status: text("status").notNull().default("pending"),
  progress: real("progress").notNull().default(0),
  filename: text("filename").notNull(),
  sampleInterval: real("sample_interval").notNull().default(1.0),
  totalFrames: integer("total_frames"),
  processedFrames: integer("processed_frames"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  completedAt: timestamp("completed_at"),
});

export const speciesDetectionsTable = pgTable("species_detections", {
  id: serial("id").primaryKey(),
  jobId: text("job_id").notNull().references(() => analysisJobsTable.id),
  species: text("species").notNull(),
  totalCount: integer("total_count").notNull().default(0),
  averageConfidence: real("average_confidence").notNull().default(0),
  color: text("color").notNull().default("#888888"),
  lastSeenAt: timestamp("last_seen_at"),
});

export const detectionFramesTable = pgTable("detection_frames", {
  id: serial("id").primaryKey(),
  jobId: text("job_id").notNull().references(() => analysisJobsTable.id),
  frameIndex: integer("frame_index").notNull(),
  timestamp: real("timestamp").notNull(),
  detections: jsonb("detections").notNull().default([]),
});

export const speciesInfoCacheTable = pgTable("species_info_cache", {
  id: serial("id").primaryKey(),
  speciesName: text("species_name").notNull().unique(),
  scientificName: text("scientific_name").notNull(),
  conservationStatus: text("conservation_status").notNull(),
  habitat: text("habitat").notNull(),
  diet: text("diet").notNull(),
  funFact: text("fun_fact").notNull(),
  description: text("description").notNull(),
  ebirdOccurrences: text("ebird_occurrences"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const insertAnalysisJobSchema = createInsertSchema(analysisJobsTable).omit({ createdAt: true });
export type InsertAnalysisJob = z.infer<typeof insertAnalysisJobSchema>;
export type AnalysisJob = typeof analysisJobsTable.$inferSelect;

export const insertSpeciesDetectionSchema = createInsertSchema(speciesDetectionsTable).omit({ id: true });
export type InsertSpeciesDetection = z.infer<typeof insertSpeciesDetectionSchema>;
export type SpeciesDetection = typeof speciesDetectionsTable.$inferSelect;

export const insertDetectionFrameSchema = createInsertSchema(detectionFramesTable).omit({ id: true });
export type InsertDetectionFrame = z.infer<typeof insertDetectionFrameSchema>;
export type DetectionFrame = typeof detectionFramesTable.$inferSelect;

export const insertSpeciesInfoCacheSchema = createInsertSchema(speciesInfoCacheTable).omit({ id: true, createdAt: true });
export type InsertSpeciesInfoCache = z.infer<typeof insertSpeciesInfoCacheSchema>;
export type SpeciesInfoCache = typeof speciesInfoCacheTable.$inferSelect;
