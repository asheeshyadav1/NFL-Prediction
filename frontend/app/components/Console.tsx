"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  compare as compareStatic,
  getAllPlayers,
  getSlate,
  getWeeks,
  recommend as recommendStatic,
} from "../lib/data";
import type {
  AllPlayer,
  PlayerCard as Card,
  Recommendation,
  Snippet,
  SlateRow,
  Week,
} from "../lib/data";

type Player = SlateRow;
type Mode = "week" | "player";

const isPlayable = (w: Week) => w.playable !== false;

// "2026-09-09" -> "Sep 9". Parsed as parts rather than through Date, which
// would shift the day for anyone west of UTC.
function openLabel(iso: string | null | undefined): string {
  if (!iso) return "";
  const [, m, d] = iso.split("-").map(Number);
  const month = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
  ][(m ?? 1) - 1];
  return month && d ? `opens ${month} ${d}` : "";
}

const POSORD: Record<string, number> = { QB: 0, RB: 1, WR: 2, TE: 3 };
const POSLABEL: Record<string, string> = {
  QB: "Quarterbacks",
  RB: "Running backs",
  WR: "Wide receivers",
  TE: "Tight ends",
};
// The matchup the artifact opened on, when the slate happens to contain it.
const OPENER = ["Ja'Marr Chase", "Justin Jefferson"];

function signed(x: number): string {
  return (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(1);
}

function Side({ p, start, peak }: { p: Card; start: boolean; peak: number }) {
  const width = (v: number) => `${Math.min(100, (v / peak) * 100).toFixed(1)}%`;
  return (
    <div className={start ? "side win" : "side"}>
      <div className="verdict">{start ? "Start him" : "Sit him"}</div>
      <div className="nm">{p.name}</div>
      <div className="mt">
        {p.position} · {p.team} vs {p.opponent}
      </div>
      <div className="pts">{p.projection.toFixed(1)}</div>
      <div className="ptl">Projected PPR points</div>
      <div className="bars">
        <div className="bar pri">
          <span>Projection</span>
          <span>{p.projection.toFixed(1)}</span>
          <div className="track">
            <i style={{ width: width(p.projection) }} />
          </div>
        </div>
        <div className="bar">
          <span>His last three games</span>
          <span>{p.baseline.toFixed(1)}</span>
          <div className="track">
            <i style={{ width: width(p.baseline) }} />
          </div>
        </div>
        <div className="bar">
          <span>Model vs. that average</span>
          <span>{signed(p.projection - p.baseline)}</span>
        </div>
        {p.career_avg !== undefined && (
          <>
            <div className="bar">
              <span>Last six games</span>
              <span>{p.last6_avg?.toFixed(1)}</span>
              <div className="track">
                <i style={{ width: width(p.last6_avg ?? 0) }} />
              </div>
            </div>
            <div className="bar">
              <span>Career ({p.career_games} games)</span>
              <span>{p.career_avg.toFixed(1)}</span>
              <div className="track">
                <i style={{ width: width(p.career_avg) }} />
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function Cite({ s }: { s: Snippet }) {
  // Anything that is not the official report is a fixture, and says so in red.
  const demo = s.source !== "NFL-INJURY-REPORT";
  return (
    <div className="cite">
      <div className="src">
        <span className={demo ? "lbl demo" : "lbl"}>{s.source}</span>
        <span>{s.player}</span>
        <span>filed {s.published}</span>
        <span>match {s.score.toFixed(2)}</span>
      </div>
      <p>{s.text}</p>
      <div className="simbar">
        <i style={{ width: `${Math.round(s.score * 100)}%` }} />
      </div>
    </div>
  );
}

function Truth({ hi, lo }: { hi: Card; lo: Card }) {
  // Completed games only. An upcoming week has no truth to show yet.
  if (hi.actual === null || lo.actual === null) {
    return (
      <div className="truthbox">
        <div className="hd">And then they actually played</div>
        <p className="fine">
          This game has not been played yet, so there is nothing to grade the call
          against. Come back when it is final.
        </p>
      </div>
    );
  }
  const hit = hi.actual >= lo.actual;
  const gap = Math.abs(hi.actual - lo.actual);
  return (
    <div className="truthbox">
      <div className="hd">And then they actually played</div>
      <div className="truth">
        {[hi, lo].map((p) => (
          <div key={p.player_id}>
            <div className="w">
              {p.name} · projected {p.projection.toFixed(1)}
            </div>
            <div className="r">
              <span className="big">{p.actual!.toFixed(1)}</span>
              <small>{signed(p.actual! - p.projection)} vs projection</small>
            </div>
          </div>
        ))}
      </div>
      <div className={hit ? "call hit" : "call miss"}>
        {hit
          ? `Good call. ${hi.name} won it by ${gap.toFixed(1)}.`
          : `Wrong call. ${lo.name} won it by ${gap.toFixed(1)}.`}
      </div>
    </div>
  );
}

export function Console() {
  const [mode, setMode] = useState<Mode>("week");
  const [allPlayers, setAllPlayers] = useState<AllPlayer[]>([]);
  const [compareA, setCompareA] = useState("");
  const [compareB, setCompareB] = useState("");
  const [weeks, setWeeks] = useState<Week[]>([]);
  const [weekKey, setWeekKey] = useState("");
  const [roster, setRoster] = useState<Player[]>([]);
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  const [result, setResult] = useState<Recommendation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [note, setNote] = useState("Loading the season");
  // "Surprise me" spans up to three renders -- new week, new slate, new pair --
  // so its intent is carried in refs rather than re-derived from state.
  const pending = useRef<{ a: string; b: string } | null>(null);
  const wantRandom = useRef(false);

  const selected = useMemo(
    () => weeks.find((w) => `${w.season}|${w.week}` === weekKey) ?? null,
    [weeks, weekKey],
  );
  // Seasons become optgroups so the option text stays "Week 15" -- the week
  // column is narrow by design and a "2024 - Week 15" label overflows it.
  const seasons = useMemo(() => {
    const out: { season: number; weeks: Week[] }[] = [];
    weeks.forEach((w) => {
      const last = out[out.length - 1];
      if (last && last.season === w.season) last.weeks.push(w);
      else out.push({ season: w.season, weeks: [w] });
    });
    return out;
  }, [weeks]);

  useEffect(() => {
    getWeeks()
      .then((all) => {
        if (all.length === 0) throw new Error("the model service has no weeks to offer");
        setWeeks(all);
        // Open on week 15 of the most recent *playable* season -- upcoming
        // weeks are listed but cannot be projected.
        const playable = all.filter(isPlayable);
        if (playable.length === 0) throw new Error("no projectable weeks yet");
        const latest = playable[playable.length - 1];
        const wanted =
          playable.find((w) => w.season === latest.season && w.week === 15) ?? latest;
        setWeekKey(`${wanted.season}|${wanted.week}`);
      })
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  // The slate reloads whenever the week does, keeping both picks if those
  // players are also playing in the new week.
  useEffect(() => {
    if (!selected) return;
    let stale = false;
    setNote("Loading the slate");
    getSlate(selected.season, selected.week)
      .then((list) => {
        if (stale) return;
        const sorted = [...list].sort(
          (x, y) =>
            (POSORD[x.position] ?? 9) - (POSORD[y.position] ?? 9) ||
            x.name.localeCompare(y.name),
        );
        setRoster(sorted);
        setError(null);
        setNote(`${sorted.length} players on this slate`);
        if (wantRandom.current && sorted.length > 1) {
          wantRandom.current = false;
          const [x, y] = pickTwo(sorted);
          pending.current = { a: x.name, b: y.name };
          setA(x.name);
          setB(y.name);
          return;
        }
        wantRandom.current = false;
        setA((prev) => keep(prev, sorted, OPENER[0], 0));
        setB((prev) => keep(prev, sorted, OPENER[1], 1));
      })
      .catch((err: unknown) => {
        if (stale) return;
        setRoster([]);
        setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      stale = true;
    };
  }, [selected]);

  // Every player who has ever been projectable. Fetched once, on demand: it is
  // far larger than a single week's slate and the weekly view never needs it.
  useEffect(() => {
    if (mode !== "player" || allPlayers.length > 0) return;
    getAllPlayers()
      .then((list) => {
        setAllPlayers(list);
        setError(null);
        setNote(`${list.length.toLocaleString()} players on record`);
        setCompareA((prev) => prev || list.find((p) => p.name === OPENER[0])?.name || list[0]?.name || "");
        setCompareB((prev) => prev || list.find((p) => p.name === OPENER[1])?.name || list[1]?.name || "");
      })
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)));
  }, [mode, allPlayers.length]);

  const compare = useCallback(async (x: string, y: string) => {
    setLoading(true);
    setError(null);
    setNote("Reading both projections, retrieving, narrating");
    const t0 = performance.now();
    try {
      setResult(await compareStatic(x, y));
      setNote(`Compared in ${Math.round(performance.now() - t0)} ms`);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "something went wrong");
      setNote("Nothing compared yet");
    } finally {
      setLoading(false);
    }
  }, []);

  const call = useCallback(
    async (playerA: string, playerB: string, season: number, week: number) => {
      setLoading(true);
      setError(null);
      setNote("Reading the projection, retrieving, narrating");
      const t0 = performance.now();
      try {
        setResult(await recommendStatic(season, week, playerA, playerB));
        setNote(`Called in ${Math.round(performance.now() - t0)} ms`);
      } catch (err) {
        setResult(null);
        setError(err instanceof Error ? err.message : "something went wrong");
        setNote("Nothing called yet");
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  // Fires the pair chosen by "Surprise me" once it is in the DOM.
  useEffect(() => {
    const p = pending.current;
    if (!p || !selected || a !== p.a || b !== p.b) return;
    pending.current = null;
    void call(p.a, p.b, selected.season, selected.week);
  }, [a, b, selected, call]);

  function surprise() {
    if (weeks.length === 0) return;
    const options = weeks.filter(isPlayable);
    if (options.length === 0) return;
    const w = options[Math.floor(Math.random() * options.length)];
    const key = `${w.season}|${w.week}`;
    if (key !== weekKey) {
      // That week's slate has not been fetched yet. Hand the pick to the
      // loader, which finishes the gesture as soon as the roster lands.
      wantRandom.current = true;
      setWeekKey(key);
      return;
    }
    if (roster.length < 2) return;
    const [x, y] = pickTwo(roster);
    pending.current = { a: x.name, b: y.name };
    setA(x.name);
    setB(y.name);
  }

  const groups = useMemo(() => {
    const out: { pos: string; players: Player[] }[] = [];
    roster.forEach((p) => {
      const last = out[out.length - 1];
      if (last && last.pos === p.position) last.players.push(p);
      else out.push({ pos: p.position, players: [p] });
    });
    return out;
  }, [roster]);

  const allGroups = useMemo(() => {
    const sorted = [...allPlayers].sort(
      (x, y) =>
        (POSORD[x.position] ?? 9) - (POSORD[y.position] ?? 9) ||
        x.name.localeCompare(y.name),
    );
    const out: { pos: string; players: AllPlayer[] }[] = [];
    sorted.forEach((p) => {
      const last = out[out.length - 1];
      if (last && last.pos === p.position) last.players.push(p);
      else out.push({ pos: p.position, players: [p] });
    });
    return out;
  }, [allPlayers]);

  function surpriseCompare() {
    if (allPlayers.length < 2) return;
    const i = Math.floor(Math.random() * allPlayers.length);
    let j = i;
    while (j === i) j = Math.floor(Math.random() * allPlayers.length);
    setCompareA(allPlayers[i].name);
    setCompareB(allPlayers[j].name);
    void compare(allPlayers[i].name, allPlayers[j].name);
  }

  const hi = result?.players.find((p) => p.name === result.start) ?? null;
  const lo = result?.players.find((p) => p.name !== result.start) ?? null;
  const peak = hi && lo
    ? Math.max(
        hi.projection, lo.projection, hi.baseline, lo.baseline,
        hi.career_avg ?? 0, lo.career_avg ?? 0,
        hi.last6_avg ?? 0, lo.last6_avg ?? 0,
        1,
      )
    : 1;

  return (
    <div id="view-app">
      <div className="wrap console">
        <div className="console-hd">
          <h2>{mode === "week" ? "Set your lineup" : "Compare players"}</h2>
          <div className="modes" role="tablist" aria-label="Comparison mode">
            <button
              role="tab"
              aria-selected={mode === "week"}
              onClick={() => { setMode("week"); setResult(null); setError(null); }}
            >
              By week
            </button>
            <button
              role="tab"
              aria-selected={mode === "player"}
              onClick={() => { setMode("player"); setResult(null); setError(null); }}
            >
              By player
            </button>
          </div>
          <div className="live">
            <i />
            <span>{note}</span>
          </div>
        </div>

        {mode === "player" ? (
          <div className="controls compare">
            {(
              [
                ["ca", "Player A", compareA, setCompareA],
                ["cb", "Player B", compareB, setCompareB],
              ] as const
            ).map(([id, label, value, set]) => (
              <div className="fld" key={id}>
                <label htmlFor={id}>{label}</label>
                <select
                  id={id}
                  value={value}
                  disabled={allPlayers.length === 0}
                  onChange={(e) => set(e.target.value)}
                >
                  {allGroups.map((g) => (
                    <optgroup key={g.pos} label={POSLABEL[g.pos] ?? g.pos}>
                      {g.players.map((p) => (
                        <option key={p.player_id} value={p.name}>
                          {p.name} · {p.team} · {p.career_avg.toFixed(1)} avg
                        </option>
                      ))}
                    </optgroup>
                  ))}
                </select>
              </div>
            ))}

            <div className="fld">
              <label>&nbsp;</label>
              <div className="btnrow">
                <button
                  className="btn sm"
                  disabled={loading || !compareA || !compareB}
                  onClick={() => compare(compareA, compareB)}
                >
                  {loading ? "Comparing…" : "Compare"}
                </button>
                <button
                  className="btn sm ghost"
                  disabled={loading || allPlayers.length < 2}
                  onClick={surpriseCompare}
                >
                  Surprise me
                </button>
              </div>
            </div>
          </div>
        ) : (
        <div className="controls">
          <div className="fld">
            <label htmlFor="wk">Week</label>
            <select
              id="wk"
              value={weekKey}
              disabled={weeks.length === 0}
              onChange={(e) => setWeekKey(e.target.value)}
            >
              {seasons.map((g) => (
                <optgroup
                  key={g.season}
                  label={
                    g.weeks.some(isPlayable)
                      ? String(g.season)
                      : `${g.season} (not played yet)`
                  }
                >
                  {g.weeks.map((w) => (
                    <option
                      key={`${w.season}|${w.week}`}
                      value={`${w.season}|${w.week}`}
                      disabled={!isPlayable(w)}
                    >
                      Week {w.week}
                      {isPlayable(w) ? "" : ` · ${openLabel(w.opens)}`}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </div>

          {(
            [
              ["pa", "Player A", a, setA],
              ["pb", "Player B", b, setB],
            ] as const
          ).map(([id, label, value, set]) => (
            <div className="fld" key={id}>
              <label htmlFor={id}>{label}</label>
              <select
                id={id}
                value={value}
                disabled={roster.length === 0}
                onChange={(e) => set(e.target.value)}
              >
                {groups.map((g) => (
                  <optgroup key={g.pos} label={POSLABEL[g.pos] ?? g.pos}>
                    {g.players.map((p) => (
                      <option key={p.player_id} value={p.name}>
                        {p.name} · {p.team} vs {p.opponent}
                      </option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </div>
          ))}

          <div className="fld">
            <label>&nbsp;</label>
            <div className="btnrow">
              <button
                className="btn sm"
                disabled={loading || !selected || !a || !b}
                onClick={() => selected && call(a, b, selected.season, selected.week)}
              >
                {loading ? "Calling…" : "Call it"}
              </button>
              <button
                className="btn sm ghost"
                disabled={loading || weeks.length === 0}
                onClick={surprise}
              >
                Surprise me
              </button>
            </div>
          </div>
        </div>
        )}

        <div className="result">
          {error && (
            <div className="fail">
              <b>No call</b>
              {error}
            </div>
          )}

          {!error && (mode === "week" ? a === b && a : compareA === compareB && compareA) && (
            <p className="fine">
              Pick two different players. The whole point is choosing between them.
            </p>
          )}

          {loading && !result && <p className="pending">Running the model…</p>}

          {result && hi && lo && (
            <>
              <div className="callline">
                Start <em>{hi.name}</em> over {lo.name}
              </div>
              <div className="callmeta">
                <span>{result.margin.toFixed(1)}-point edge</span>
                <span className={`chip ${result.confidence}`}>
                  {result.confidence} confidence
                </span>
                <span>
                  {mode === "player"
                    ? `latest form · ${hi.season} wk${hi.week} vs ${lo.season} wk${lo.week}`
                    : `${hi.season} · week ${hi.week}`}
                </span>
              </div>

              <div className="matchup">
                <Side p={hi} start peak={peak} />
                <Side p={lo} start={false} peak={peak} />
              </div>

              <div className="blk">
                <div className="hd">
                  <span>What it says</span>
                  {result.narration_grounded ? (
                    <span className="tag">Numbers check out</span>
                  ) : (
                    <span className="tag bad">Numbers do not check out</span>
                  )}
                </div>
                <div className="say">{result.narration}</div>
                <div className="fine">
                  Narrated by {result.narration_model}.{" "}
                  {result.narration_grounded
                    ? "Every point total quoted above is the model's own, matched digit for digit before this was shown to you."
                    : "The write-up quoted a total the model never produced. Trust the prose, not the numbers."}
                </div>
              </div>

              <div className="blk">
                <div className="hd">
                  <span>What it read first</span>
                  <span>{result.snippets.length} from the injury report</span>
                </div>
                {result.snippets.length > 0 ? (
                  <div className="cites">
                    {result.snippets.map((s, i) => (
                      <Cite key={`${s.source}-${s.player}-${i}`} s={s} />
                    ))}
                  </div>
                ) : (
                  <p className="fine">Nothing on either player this week.</p>
                )}
              </div>

              <Truth hi={hi} lo={lo} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// Hold a selection across a week change when that player is on the new slate,
// otherwise fall back to the artifact's opener, otherwise the top of the list.
function keep(prev: string, roster: Player[], opener: string, index: number): string {
  if (prev && roster.some((p) => p.name === prev)) return prev;
  if (roster.some((p) => p.name === opener)) return opener;
  return roster[index]?.name ?? roster[0]?.name ?? "";
}

// Two distinct players from the slate, in list order for a stable A/B.
function pickTwo(roster: Player[]): [Player, Player] {
  const i = Math.floor(Math.random() * roster.length);
  let j = i;
  while (j === i) j = Math.floor(Math.random() * roster.length);
  return i < j ? [roster[i], roster[j]] : [roster[j], roster[i]];
}
