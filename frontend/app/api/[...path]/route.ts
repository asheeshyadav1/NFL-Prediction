import { NextRequest } from "next/server";

// Proxies the browser's same-origin /api/* calls to the gateway.
//
// A route handler rather than a `rewrites()` entry: Next bakes rewrite
// destinations in at build time, which would freeze API_URL into the image.
// Reading it here resolves it per request. In Kubernetes the ingress routes
// /api before Next sees it, so this is normally unused there.

export const dynamic = "force-dynamic";

const HOP_BY_HOP = new Set([
  "connection", "keep-alive", "transfer-encoding", "upgrade", "host",
]);

async function proxy(
  req: NextRequest,
  { params }: { params: Promise<{ path: string[] }> },
) {
  const base = (process.env.API_URL ?? "http://localhost:8000").replace(/\/$/, "");
  const { path } = await params;
  const url = `${base}/${path.join("/")}${req.nextUrl.search}`;

  const headers = new Headers();
  req.headers.forEach((value, key) => {
    if (!HOP_BY_HOP.has(key.toLowerCase())) headers.set(key, value);
  });

  try {
    const upstream = await fetch(url, {
      method: req.method,
      headers,
      body: ["GET", "HEAD"].includes(req.method) ? undefined : await req.text(),
      cache: "no-store",
    });
    return new Response(upstream.body, {
      status: upstream.status,
      headers: {
        "content-type": upstream.headers.get("content-type") ?? "application/json",
      },
    });
  } catch (err) {
    // A down gateway is a 502 from the proxy, not an opaque client-side error.
    return Response.json(
      { detail: `gateway unreachable: ${err instanceof Error ? err.message : err}` },
      { status: 502 },
    );
  }
}

export { proxy as GET, proxy as POST };
