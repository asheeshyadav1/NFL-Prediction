import type { NextConfig } from "next";

// No `rewrites()` or `env` block on purpose: Next resolves both at build time,
// baking an API address into the image. app/api/[...path]/route.ts proxies
// /api/* instead, reading API_URL per request.
const nextConfig: NextConfig = {
  // "standalone" produces the small runtime image Dockerfile.frontend wants.
  // Netlify's Next runtime does its own packaging and trips over it, so opt out
  // there rather than carrying two config files.
  ...(process.env.NETLIFY ? {} : { output: "standalone" as const }),
};

export default nextConfig;
