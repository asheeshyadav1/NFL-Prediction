"""Ingest preseason signal into the retrieval corpus.

    python scripts/ingest_preseason.py --season 2026

Before week 1 there are no box scores and no official injury report, so the
injury ingest has nothing to say about the coming season. Two feeds do exist
and do move: the team depth chart (re-snapshotted through camp) and the weekly
roster (active / reserve / cut). Together they answer the only questions that
can be asked before kickoff -- did he make the roster, and is he the starter.

Writes `data/news/preseason.json`. Corpora are additive and each keeps its own
`source` label, so these stay distinguishable from the injury report at
citation time. Re-running overwrites by id.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

from data import load_depth_charts, load_rosters, load_schedules  # noqa: E402

log = logging.getLogger("ingest-preseason")

OUT_PATH = ROOT / "data" / "news" / "preseason.json"
SOURCE = "NFL-DEPTH-CHART"
POSITIONS = ("QB", "RB", "WR", "TE")

# Roster status codes, spelled out. Anything not here is reported by its code
# rather than guessed at.
STATUS_TEXT = {
    "ACT": "on the active roster",
    "RES": "on reserve (injured reserve or similar) and unavailable",
    "CUT": "no longer on the roster",
    "EXE": "on the exempt list",
    "RET": "retired",
    "PUP": "on the physically-unable-to-perform list",
    "NON": "on the non-football injury list",
}

# How a depth-chart rank reads in a sentence.
RANK_TEXT = {1: "first-string", 2: "second-string", 3: "third-string"}


def _week_of(dt: pd.Series, schedule: pd.DataFrame) -> pd.Series:
    """Map a snapshot timestamp to the season week it describes.

    A chart published before the season opens describes week 1; after that it
    describes the next week that has not yet kicked off.
    """
    if schedule.empty:
        return pd.Series(1, index=dt.index)
    starts = (
        schedule.groupby("week")["gameday"].min().sort_index().astype("datetime64[ns]")
    )
    stamps = pd.to_datetime(dt, utc=True, errors="coerce").dt.tz_localize(None)
    weeks = [
        int(starts.index[starts <= s].max()) if (starts <= s).any() else 1
        for s in stamps
    ]
    return pd.Series(weeks, index=dt.index)


def latest_depth_chart(charts: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """The most recent snapshot per team, offensive skill positions only."""
    if charts.empty:
        return charts
    df = charts[charts["pos_abb"].isin(POSITIONS)].copy()
    if df.empty:
        return df
    # One chart per team: the newest the feed has for that team.
    newest = df.groupby("team")["dt"].transform("max")
    df = df[df["dt"] == newest].copy()
    df["week"] = _week_of(df["dt"], schedule)
    df["published"] = (
        pd.to_datetime(df["dt"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    )
    return df


def describe(row: pd.Series, status: str | None) -> str:
    """One depth-chart row as a sentence."""
    rank = int(row["pos_rank"]) if pd.notna(row["pos_rank"]) else None
    place = RANK_TEXT.get(rank, f"number {rank}" if rank else "listed")
    parts = [
        f"{row['player_name']} is the {place} {row['pos_abb']} on {row['team']}'s "
        f"depth chart for week {int(row['week'])} of the {int(row['season'])} season."
    ]
    if status:
        parts.append(
            f"Roster status: {STATUS_TEXT.get(status, status)}."
            if status in STATUS_TEXT
            else f"Roster status: {status}."
        )
    parts.append("No games have been played yet; this is preseason positioning.")
    return " ".join(parts)


def build_snippets(season: int) -> list[dict]:
    charts = load_depth_charts([season])
    if charts.empty:
        log.error("no depth chart release for %s", season)
        return []
    charts["season"] = season

    schedule = load_schedules([season])
    chart = latest_depth_chart(charts, schedule)
    if chart.empty:
        log.error("no offensive depth-chart rows for %s", season)
        return []

    # Roster status, joined on gsis_id where the feed has one.
    rosters = load_rosters([season])
    status_by_id: dict[str, str] = {}
    if not rosters.empty and "gsis_id" in rosters.columns:
        latest_wk = rosters["week"].max() if "week" in rosters.columns else None
        recent = rosters[rosters["week"] == latest_wk] if latest_wk is not None else rosters
        status_by_id = {
            str(r["gsis_id"]): str(r["status"])
            for _, r in recent.iterrows()
            if pd.notna(r.get("gsis_id")) and pd.notna(r.get("status"))
        }

    records = []
    for _, r in chart.iterrows():
        gsis = str(r["gsis_id"]) if pd.notna(r.get("gsis_id")) else None
        if not gsis or pd.isna(r.get("published")):
            continue
        records.append(
            {
                "id": f"dc-{season}-{int(r['week']):02d}-{gsis}",
                "player": r["player_name"],
                "team": r["team"],
                "published": r["published"],
                "season": season,
                "week": int(r["week"]),
                "source": SOURCE,
                "text": describe(r, status_by_id.get(gsis)),
            }
        )
    records.sort(key=lambda s: (s["published"], s["id"]), reverse=True)
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    snippets = build_snippets(args.season)
    if not snippets:
        log.error("nothing to write for %s", args.season)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "_README": (
            "REAL DATA. Generated by scripts/ingest_preseason.py from the nflverse "
            "`depth_charts` and `weekly_rosters` releases. Preseason positioning "
            "only -- no games have been played. Regenerate rather than hand-editing."
        ),
        "_schema": "id, player, team, published, season, week, source, text",
        "_source_label": f"Every entry carries source = '{SOURCE}'.",
        "snippets": snippets,
    }, indent=2) + "\n")

    log.info("wrote %d snippets to %s (%d players, %d teams, snapshot %s)",
             len(snippets), args.out, len({s["player"] for s in snippets}),
             len({s["team"] for s in snippets}), max(s["published"] for s in snippets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
