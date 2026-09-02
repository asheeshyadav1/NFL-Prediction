// Narration, and only narration.
//
// Everything else the app needs is precomputed into public/data by
// scripts/export_static.py. The write-up cannot be: it is asked about a pair the
// visitor chooses at runtime. It is a single fetch with no dependencies, so it
// fits a function where the Python service did not.
//
// The key stays here. It is read from the environment on the server side and is
// never sent to the browser.

import type { Config, Context } from "@netlify/functions";

const MODEL = process.env.GEMINI_MODEL || "gemini-3.6-flash";

// Same contract as api/llm.py: the projections are facts, the model explains
// them and is not allowed to produce numbers of its own.
const SYSTEM = `You explain fantasy football start/sit recommendations.

A trained sequence model produced the projected point totals. They are given to you as facts. Your job is to explain the recommendation, not to make one of your own.

Rules, in priority order:
1. Never invent, recompute, adjust, or second-guess a projected point total. Quote the numbers you are given, exactly as given.
2. Reason only from the projections and the retrieved snippets below. If the snippets do not mention a player, say the context is thin rather than speculating about their situation.
3. The snippets are the only news you have. Do not add injury, depth-chart, or matchup claims from your own knowledge, you have no way to know whether they are current.
4. If a snippet cuts against the projection (an injury designation on the higher projected player, say), surface that tension explicitly. The projection still stands; the reader deserves to know the risk.

Write 2-4 sentences. Open with the recommendation. Plain prose, no headings, no bullet points. Never use em dashes; use a comma, colon or full stop instead.`;

type Player = {
  name: string; position: string; team: string; opponent: string; projection: number;
};

// Belt and braces: the prompt asks for no em dashes, but a prompt is a request.
const stripDashes = (s: string) =>
  s.replace(/ [—–] /g, ", ").replace(/[—–]/g, ", ");

export default async (req: Request, _context: Context) => {
  if (req.method !== "POST") return new Response("method not allowed", { status: 405 });

  const key = process.env.GEMINI_API_KEY;
  // No key configured is a normal state, not an error: the caller falls back to
  // the deterministic template.
  if (!key) return Response.json({ text: null, model: "template (no key)" }, { status: 200 });

  let body: { a?: Player; b?: Player; snippets?: { source: string; published: string; player: string; text: string }[] };
  try {
    body = await req.json();
  } catch {
    return Response.json({ error: "bad request" }, { status: 400 });
  }
  const { a, b, snippets = [] } = body;
  if (!a || !b) return Response.json({ error: "two players required" }, { status: 400 });

  const context = snippets.length
    ? snippets.map((s) => `- [${s.source} | ${s.published}] ${s.player}: ${s.text}`).join("\n")
    : "- (no relevant snippets retrieved)";

  const prompt = `Compare these two players for the upcoming week.

PROJECTIONS (from the trained model, treat as fact):
- ${a.name} (${a.position}, ${a.team}) vs ${a.opponent}: ${a.projection.toFixed(1)} PPR points
- ${b.name} (${b.position}, ${b.team}) vs ${b.opponent}: ${b.projection.toFixed(1)} PPR points

RETRIEVED NEWS SNIPPETS:
${context}

Which should the manager start, and why?`;

  try {
    const res = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-goog-api-key": key },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }],
          systemInstruction: { parts: [{ text: SYSTEM }] },
          generationConfig: { temperature: 0.3, maxOutputTokens: 1024 },
        }),
      },
    );
    if (!res.ok) {
      // 503s from an overloaded model are routine; the template covers them.
      return Response.json({ text: null, model: `template (narrator ${res.status})` });
    }
    const data = await res.json();
    const text = data?.candidates?.[0]?.content?.parts?.map((p: { text?: string }) => p.text ?? "").join("").trim();
    if (!text) return Response.json({ text: null, model: "template (model declined)" });
    return Response.json({ text: stripDashes(text), model: MODEL });
  } catch {
    return Response.json({ text: null, model: "template (narrator unavailable)" });
  }
};

export const config: Config = { path: "/api/narrate" };
