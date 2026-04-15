# EcoBird Counter

## Overview

EcoBird Counter is a real-time AI bird monitoring web application. Users upload video files (MP4/AVI/MOV), configure a sampling interval, and the system simulates frame-by-frame bird detection — identifying species, counting individuals, and preventing double-counting via track IDs. An AI agent (OpenAI GPT) generates detailed species info cards with scientific data, conservation status, habitat, diet, and fun facts.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **Frontend**: React 19 + Vite + TailwindCSS v4 + shadcn/ui
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **AI**: OpenAI GPT-5.2 (via Replit AI Integrations) for species info cards
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

1. **Video Upload & Analysis** — Upload video, set sample interval (0.5–5s), triggers a simulated frame-by-frame detection pipeline
2. **Species Detection** — 8 species: Sparrow (blue), Finch (green), Warbler (red), Robin (yellow), Kingfisher (cyan), Swallow (purple), Eagle (orange), Unknown (gray)
3. **AI Species Info Cards** — Click any species to trigger GPT-5.2 agent, generates a Markdown-style info card with scientific name, conservation status, habitat, diet, fun fact, and North African eBird context. Results are cached in the DB.
4. **Analysis History** — View past analysis jobs with status, progress, and species counts
5. **Statistics Dashboard** — Global aggregates: total analyses, total detections, top species, recent activity

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
