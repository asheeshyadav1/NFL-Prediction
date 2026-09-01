import { NextRequest } from "next/server";

// Proxies the browser's same-origin /api/* calls to the gateway.
//
// This is a route handler rather than a `rewrites()` entry in next.config.ts on
// purpose: Next bakes rewrite destinations into routes-manifest.json at build
// time, so a rewrite would freeze whatever API_URL happened to be set during
// `next build` and silently ignore the value the deployment supplies. Reading
// the variable inside the handler resolves it per request, which is what lets
// one image run against a different gateway in compose and in Kubernetes.
//
// In Kubernetes this code is normally never reached: the ingress routes /api
// straight to the API service before it gets to Next.

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
