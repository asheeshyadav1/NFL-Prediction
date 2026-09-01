import type { NextConfig } from "next";

// The browser calls same-origin /api/*. Under the ingress that prefix goes
// straight to the API service; otherwise app/api/[...path]/route.ts proxies it
// there, reading API_URL per request. Deliberately no `rewrites()` entry and no
// `env` block here -- Next resolves both at build time, which would bake an
// address into the image and ignore what the deployment sets.
const nextConfig: NextConfig = {
  output: "standalone", // smallest possible container image
};

export default nextConfig;
