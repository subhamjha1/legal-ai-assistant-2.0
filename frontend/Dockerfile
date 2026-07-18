# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1: deps - install dependencies in their own layer so `npm install`
# is only re-run when package*.json actually changes, not on every source
# code edit (Docker layer caching).
# ---------------------------------------------------------------------------
FROM node:20-slim AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci

# ---------------------------------------------------------------------------
# Stage 2: builder - compiles the production build.
# ---------------------------------------------------------------------------
FROM node:20-slim AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# NEXT_PUBLIC_* variables are inlined into the client bundle at BUILD time,
# not read at container start - so this must be a build ARG passed via
# docker-compose's `build.args`, not just a runtime `environment:` entry.
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

RUN npm run build

# ---------------------------------------------------------------------------
# Stage 3: runner - minimal runtime image. Uses Next's "standalone" output
# (next.config.ts: output: "standalone"), which traces and copies only the
# node_modules actually needed at runtime - dramatically smaller than
# shipping the full builder stage's node_modules.
# ---------------------------------------------------------------------------
FROM node:20-slim AS runner
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 nodeuser \
    && useradd --uid 1000 --gid nodeuser --shell /bin/bash --create-home nodeuser

ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME=0.0.0.0

COPY --from=builder --chown=nodeuser:nodeuser /app/.next/standalone ./
COPY --from=builder --chown=nodeuser:nodeuser /app/.next/static ./.next/static
COPY --from=builder --chown=nodeuser:nodeuser /app/public ./public

USER nodeuser

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:3000/api/health || exit 1

CMD ["node", "server.js"]
