"use client";

import { useState } from "react";

const API = process.env.API_URL ?? "http://localhost:8000";

type Projection = {
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
};

type Snippet = {
  player: string;
  published: string;
  source: string;
  text: string;
  score: number;
};

type Recommendation = {
  players: Projection[];
  start: string;
  margin: number;
  confidence: "high" | "moderate" | "low";
  snippets: Snippet[];
  narration: string;
  narration_model: string;
  narration_grounded: boolean;
};

function PlayerCard({ p, winner }: { p: Projection; winner: boolean }) {
  const delta = p.projection - p.baseline;
  return (
    <div className={winner ? "card winner" : "card"}>
      <div className="tag">{winner ? "Start" : "Sit"}</div>
      <h2>{p.name}</h2>
      <div className="meta">
        {p.position} · {p.team} vs {p.opponent} · {p.season} wk {p.week}
      </div>
      <div className="proj">
        {p.projection.toFixed(1)}
        <span>projected PPR</span>
      </div>
      <div className="compare">
        <span>
          Naive last-3 baseline: {p.baseline.toFixed(1)} ({delta >= 0 ? "+" : ""}
          {delta.toFixed(1)})
        </span>
        {p.actual !== null && <span>Actual: {p.actual.toFixed(1)}</span>}
      </div>
    </div>
  );
}

export default function Page() {
  const [playerA, setPlayerA] = useState("Mark Andrews");
  const [playerB, setPlayerB] = useState("Brock Bowers");
  const [result, setResult] = useState<Recommendation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function compare(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API}/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ player_a: playerA, player_b: playerB }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body.detail ?? `request failed (${res.status})`);
      setResult(body as Recommendation);
    } catch (err) {
      setError(err instanceof Error ? err.message : "something went wrong");
    } finally {
      setLoading(false);
    }
  }

  const winner = result?.players.find((p) => p.name === result.start);

  return (
    <main>
      <header>
        <h1>Fantasy Football Toolkit</h1>
        <p>
          The projection comes from a trained sequence model — not from the
          language model. The LLM only explains the number, grounded in
          retrieved news. Numbers below are for a completed week, so the actual
          result is shown alongside for honesty.
        </p>
      </header>

      <form onSubmit={compare}>
        <div>
          <label htmlFor="a">Player A</label>
          <input id="a" value={playerA} onChange={(e) => setPlayerA(e.target.value)} />
        </div>
        <div>
          <label htmlFor="b">Player B</label>
          <input id="b" value={playerB} onChange={(e) => setPlayerB(e.target.value)} />
        </div>
        <button type="submit" disabled={loading}>
          {loading ? "Projecting…" : "Compare"}
        </button>
      </form>

      {error && <div className="error">{error}</div>}

      {result && winner && (
        <>
          <div className="cards">
            {result.players.map((p) => (
              <PlayerCard key={p.player_id} p={p} winner={p.name === result.start} />
            ))}
          </div>

          <div className="panel">
            <h3>
              Recommendation · {result.margin.toFixed(1)}-point edge ·{" "}
              {result.confidence} confidence
            </h3>
            <div className="narration">{result.narration}</div>
            <div className="footnote">
              Narrated by {result.narration_model}.
              {result.narration_grounded
                ? " Quoted projections verified against the model output."
                : ""}
            </div>
            {!result.narration_grounded && (
              <div className="warnbar">
                Warning: the narration did not quote the model&apos;s projected
                totals verbatim — treat the prose, not the numbers, as suspect.
              </div>
            )}
          </div>

          <div className="panel">
            <h3>Retrieved context ({result.snippets.length})</h3>
            {result.snippets.length === 0 && (
              <div className="snippet">Nothing relevant was retrieved.</div>
            )}
            {result.snippets.map((s, i) => (
              <div className="snippet" key={i}>
                <div className="src">
                  {s.player} · {s.source} · {s.published} · similarity{" "}
                  {s.score.toFixed(2)}
                </div>
                {s.text}
              </div>
            ))}
          </div>
        </>
      )}
    </main>
  );
}
