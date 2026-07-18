import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Standalone output produces a self-contained server bundle (node_modules
  // trimmed to only what's actually needed at runtime) - this is what lets
  // the production Docker image copy just .next/standalone instead of the
  // full node_modules tree, keeping the final image small.
  output: "standalone",
};

export default nextConfig;
