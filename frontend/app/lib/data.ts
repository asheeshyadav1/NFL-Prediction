// Static data layer.
//
// The Python service is gone: it wanted torch, ~2GB and a machine that never
// sleeps. Every answer it gave was deterministic, so scripts/export_static.py
// runs the model once and writes the results under public/data. This module
// loads those files and applies the same decision rule the gateway applied.
//
// The rule lives in one place here, exactly as `_decide()` did server-side, so
// the two views cannot drift apart.

export type Week = {
  season: number;
  week: number;
  player_weeks: number;
  playable?: boolean;
  opens?: string | null;
};

export type SlateRow = {
  player_id: string;
  name: string;
  position: string;
  team: string;
  opponent: string;
  season: number;
  week: number;
  projection: number;
  baseline: number;
  actual: number | null;
  cites: string[];
};

export type PlayerCard = SlateRow & {
  career_games?: number;
  career_avg?: number;
  last6_avg?: number;
};

export type AllPlayer = {
  player_id: string;
  name: string;
  position: string;
  team: string;
  career_games: number;
  career_avg: number;
};

export type Snippet = {
  player: string;
  published: string;
  source: string;
  text: string;
  score: number;
};

export type Recommendation = {
  players: PlayerCard[];
  start: string;
  margin: number;
  confidence: "high" | "moderate" | "low";
  snippets: Snippet[];
  narration: string;
  narration_model: string;
  narration_grounded: boolean;
};

const BASE = "/data";

// Each file is fetched at most once per page load.
const cache = new Map<string, Promise<unknown>>();

function load<T>(path: string): Promise<T> {
  let hit = cache.get(path);
  if (!hit) {
    hit = fetch(`${BASE}/${path}`).then((r) => {
      if (!r.ok) throw new Error(`could not load ${path} (${r.status})`);
      return r.json();
    });
    cache.set(path, hit);
  }
  return hit as Promise<T>;
}

export const getWeeks = () => load<Week[]>("weeks.json");
export const getAllPlayers = () => load<AllPlayer[]>("players.json");
export const getSlate = (season: number, week: number) =>
  load<SlateRow[]>(`slates/${season}-${week}.json`);

type SnippetTable = Record<string, Omit<Snippet, "score">>;
const getSnippets = () => load<SnippetTable>("snippets.json");
const getLatest = () => load<Record<string, PlayerCard>>("latest.json");

function confidenceOf(margin: number): Recommendation["confidence"] {
  return margin >= 3 ? "high" : margin >= 1 ? "moderate" : "low";
}

// Mirrors api/llm.py's template, including its punctuation.
function template(hi: PlayerCard, lo: PlayerCard, margin: number, cite: string | null): string {
  const conf =
    margin >= 3 ? "clear" : margin >= 1 ? "narrow" : "effectively a coin flip";
  const head =
    `Start ${hi.name} over ${lo.name}. The model projects ` +
    `${hi.projection.toFixed(1)} PPR points for ${hi.name} against ${hi.opponent}, ` +
    `versus ${lo.projection.toFixed(1)} for ${lo.name} against ${lo.opponent}, ` +
    `a ${margin.toFixed(1)}-point edge, which is ${conf}.`;
  return cite
    ? `${head} Retrieved context: ${cite}`
    : `${head} No relevant news was retrieved, so this rests on the projection alone.`;
}

// The same check the service ran: did the prose quote our numbers, or its own?
function grounded(text: string, players: PlayerCard[]): boolean {
  return players.every((p) => text.includes(p.projection.toFixed(1)));
}

async function resolveCites(rows: PlayerCard[]): Promise<Snippet[]> {
  const ids = [...new Set(rows.flatMap((r) => r.cites ?? []))];
  if (ids.length === 0) return [];
  const table = await getSnippets();
  // Score is not recomputed in the browser; ordering was fixed at export time
  // by the same store the service used, so the ranking is already the service's.
  return ids
    .map((id, i) => (table[id] ? { ...table[id], score: 1 - i * 0.01 } : null))
    .filter((s): s is Snippet => s !== null);
}

async function narrate(
  hi: PlayerCard, lo: PlayerCard, margin: number, snippets: Snippet[],
): Promise<Pick<Recommendation, "narration" | "narration_model" | "narration_grounded">> {
  const cite = snippets.length
    ? `[${snippets[0].source} | ${snippets[0].published}] ${snippets[0].player}: ${snippets[0].text}`
    : null;
  const fallback = template(hi, lo, margin, cite);

  // The narrator is the one thing that cannot be precomputed, since it is asked
  // about a pair chosen at runtime. It runs as a Netlify function; if that is
  // unconfigured or down, the deterministic template stands in, exactly as it
  // did server-side.
  try {
    const res = await fetch("/api/narrate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ a: hi, b: lo, snippets: snippets.slice(0, 4) }),
    });
    if (res.ok) {
      const body = await res.json();
      if (typeof body?.text === "string" && body.text.trim()) {
        return {
          narration: body.text,
          narration_model: body.model ?? "gemini",
          narration_grounded: grounded(body.text, [hi, lo]),
        };
      }
    }
  } catch {
    /* fall through to the template */
  }
  return {
    narration: fallback,
    narration_model: "template (no narrator configured)",
    narration_grounded: true,
  };
}

async function decide(a: PlayerCard, b: PlayerCard): Promise<Recommendation> {
  if (a.player_id === b.player_id) throw new Error("pick two different players");
  const [hi, lo] = a.projection >= b.projection ? [a, b] : [b, a];
  const margin = Math.round((hi.projection - lo.projection) * 10) / 10;
  const snippets = await resolveCites([a, b]);
  return {
    players: [a, b],
    start: hi.name,
    margin,
    confidence: confidenceOf(margin),
    snippets,
    ...(await narrate(hi, lo, margin, snippets)),
  };
}

export async function recommend(
  season: number, week: number, nameA: string, nameB: string,
): Promise<Recommendation> {
  const slate = await getSlate(season, week);
  const find = (n: string) => {
    const row = slate.find((r) => r.name === n);
    if (!row) throw new Error(`no projectable player matching ${n} in ${season} week ${week}`);
    return row;
  };
  return decide(find(nameA), find(nameB));
}

export async function compare(nameA: string, nameB: string): Promise<Recommendation> {
  const [latest, players] = await Promise.all([getLatest(), getAllPlayers()]);
  const find = (n: string) => {
    const p = players.find((x) => x.name === n);
    const row = p ? latest[p.player_id] : undefined;
    if (!row) throw new Error(`no projectable player matching ${n}`);
    return row;
  };
  return decide(find(nameA), find(nameB));
}
