import { NextResponse } from "next/server";

/**
 * Lightweight liveness check for container orchestration (Docker
 * healthcheck, Render/Railway health checks, load balancer probes). Does
 * NOT check backend connectivity - that's a separate concern (a failed
 * backend shouldn't necessarily take the frontend container out of
 * rotation, since the UI can still render its empty/error states
 * gracefully) - this only confirms the Next.js server process itself is
 * up and serving requests.
 */
export async function GET() {
  return NextResponse.json({ status: "ok", service: "legal-ai-assistant-frontend" });
}
