# EcoBird Counter

## Overview

EcoBird Counter is a production-grade AI bird monitoring web application. Users upload video files (MP4/AVI/MOV), configure a sampling interval, and a multi-stage computer vision + AI pipeline performs frame-by-frame bird detection — identifying species, tracking individuals with stable IDs, and counting unique birds across frames without inflation. Focused on North African and Algerian biodiversity.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **Frontend**: React 19 + Vite + TailwindCSS v4 + shadcn/ui
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **AI Detection**: Google Gemini 1.5 Flash — per-frame bird detection
- **Tracking**: ByteTracker (IoU-based, TypeScript) — stable track IDs across frames
- **Open-Set Recovery**: Gemini visual reasoning pass for low-confidence detections (<0.60)
- **Species Validation**: Wikipedia REST API cross-reference
- **Validation**: Zod (zod/v4), drizzle-zod
- **API codegen**: Orval (from OpenAPI spec)
- **File upload**: Multer (multipart/form-data)
- **Build**: esbuild (CJS bundle for API), Vite (frontend)

## Architecture

- `artifacts/ecobird-counter/` — React + Vite frontend (served at `/`)
- `artifacts/api-server/` — Express API server (served at `/api`)
- `lib/db/` — Drizzle ORM + PostgreSQL schema
- `lib/api-spec/openapi.yaml` — OpenAPI contract
- `lib/api-client-react/` — Generated React Query hooks (from Orval)
- `lib/api-zod/` — Generated Zod validation schemas (from Orval)
- `lib/integrations-openai-ai-server/` — OpenAI server-side integration

## Key Features

1. **Video Upload & Analysis** — Upload video (MP4/AVI/MOV up to 500MB), set sample interval (0.5–10s), triggers the multi-stage detection pipeline.
2. **Gemini 1.5 Flash Detection** — Per-frame bird detection with a rigorous ornithology prompt; high-res 720px frames for occlusion penetration.
3. **ByteTracker** — IoU-based track assignment (`artifacts/api-server/src/lib/bytetrack.ts`) maintains globally-consistent track IDs across all frames, preventing ID switching and count inflation.
4. **Open-Set Recovery** — Any detection with confidence < 0.60 triggers a dedicated Gemini visual reasoning pass to either confirm, relabel, or discard false positives.
5. **Wikipedia Validation** — Species names are cross-referenced against Wikipedia REST API after tracking; unvalidated labels are logged but kept.
6. **Unique-bird Counting** — Species counts represent unique track IDs, not raw detection totals — eliminating count inflation from the same bird appearing in multiple frames.
7. **Parallel Processing** — Frames extracted concurrently (4 ffmpeg workers); Gemini calls run 2 in parallel with exponential-backoff retry.
8. **AI Species Info Cards** — Click any species to trigger Gemini 1.5 Flash, generates a card with scientific name, conservation status, habitat, diet, fun fact, and North African eBird context. Results cached in DB.
9. **Analysis History** — View past jobs with progress tracking.
10. **Statistics Dashboard** — Global aggregates: total analyses, unique birds, top species.

## Database Tables

- `analysis_jobs` — Analysis job records (status, progress, filename, interval)
- `species_detections` — Per-job species detection results
- `detection_frames` — Per-frame detection data (bounding boxes, track IDs)
- `species_info_cache` — Cached AI-generated species info cards

## Key Commands

- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- `pnpm --filter @workspace/api-server run dev` — run API server locally

## Environment Variables

- `DATABASE_URL` — PostgreSQL connection string (auto-set by Replit)
- `AI_INTEGRATIONS_OPENAI_BASE_URL` — OpenAI proxy URL (auto-set by Replit AI Integrations)
- `AI_INTEGRATIONS_OPENAI_API_KEY` — OpenAI proxy key (auto-set by Replit AI Integrations)
- `SESSION_SECRET` — Session secret (pre-configured)
