import type { NextConfig } from "next";

// No `rewrites()` or `env` block on purpose: Next resolves both at build time,
// baking an API address into the image. app/api/[...path]/route.ts proxies
// /api/* instead, reading API_URL per request.
const nextConfig: NextConfig = {
  output: "standalone", // smallest possible container image
};

export default nextConfig;
