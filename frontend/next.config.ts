import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone", // smallest possible container image
  env: {
    API_URL: process.env.API_URL ?? "http://localhost:8000",
  },
};

export default nextConfig;
