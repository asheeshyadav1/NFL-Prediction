"""Raw data pull from nflverse, with on-disk caching.

Fetch and cache only, so the rest of the pipeline is deterministic and runs
offline once warm. Reads the release files directly rather than via
`nfl_data_py`, which pins `pandas<2` and conflicts with a current torch stack.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
import urllib.request

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# nflverse moved weekly player stats out of the `player_stats` release and into
# `stats_player`. The old release is frozen at 2024, so the current release has
# to be tried first or every season from 2025 on silently looks unavailable.
_RELEASE = "https://github.com/nflverse/nflverse-data/releases/download/stats_player"
_LEGACY_RELEASE = "https://github.com/nflverse/nflverse-data/releases/download/player_stats"
# Names churned twice; a few seasons (2019 at time of writing) only exist under
# the oldest one, so try each in turn.
WEEKLY_URLS = (
    _RELEASE + "/stats_player_week_{year}.parquet",
    _LEGACY_RELEASE + "/stats_player_week_{year}.parquet",
    _LEGACY_RELEASE + "/player_stats_{year}.parquet",
)
GAMES_URL = "http://www.habitatring.com/games.csv"

# Preseason signal. Before week 1 there are no box scores and no injury report,
# but both of these exist and move: the depth chart is re-snapshotted through
# camp, and the weekly roster carries cut/reserve/active status.
DEPTH_CHARTS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_{year}.parquet"
)
ROSTERS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_{year}.parquet"
)

# The official NFL injury report (the nfl.com/injuries data), mirrored by
# nflverse. More stable than that page's token-gated API, and it carries gsis_id.
INJURIES_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{year}.parquet"
)

# Kickers and team defenses score differently and would need their own model.
POSITIONS = ("QB", "RB", "WR", "TE")

# Renames to a stable internal schema, so a change upstream is a one-line fix.
COLUMN_ALIASES = {
    "team": "recent_team",
    "passing_interceptions": "interceptions",
}

# The schedule feed keeps a franchise's historical abbreviation; the stats feed
# backfills the current one. Without this the relocated franchises fail to join.
TEAM_ALIASES = {"SD": "LAC", "OAK": "LV", "STL": "LA"}

# Enough of the schema to build an empty frame with the expected columns.
INJURY_COLUMNS = (
    "season", "week", "team", "game_type", "gsis_id", "position", "full_name",
    "report_primary_injury", "report_secondary_injury", "report_status",
    "practice_primary_injury", "practice_status",
)


def _download(url: str) -> bytes:
    log.info("fetching %s", url)
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()


def _download_first(urls: tuple[str, ...], **fmt) -> bytes:
    last: Exception | None = None
    for url in urls:
        try:
            return _download(url.format(**fmt))
        except urllib.error.HTTPError as exc:  # try the next naming convention
            last = exc
    raise RuntimeError(f"no nflverse asset found for {fmt}") from last


def load_weekly(years: list[int]) -> pd.DataFrame:
    """Per-player, per-week box scores including precomputed PPR points.

    A season with no release yet is skipped with a warning rather than raising,
    so a range that runs into the current season works before week 1. The
    seasons actually loaded are logged -- never assume you got what you asked
    for, check the range in the log.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    missing = []
    for year in years:
        path = CACHE_DIR / f"weekly_{year}.parquet"
        if not path.exists():
            try:
                path.write_bytes(_download_first(WEEKLY_URLS, year=year))
            except RuntimeError:
                log.warning("no weekly release for %s yet -- skipping", year)
                missing.append(year)
                continue
        season = pd.read_parquet(path)
        season = season.rename(
            columns={k: v for k, v in COLUMN_ALIASES.items() if k in season.columns}
        )
        frames.append(season)

    if not frames:
        raise RuntimeError(f"no weekly data available for any of {years}")

    df = pd.concat(frames, ignore_index=True)
    df = df[df["season_type"] == "REG"]
    df = df[df["position"].isin(POSITIONS)]
    log.info(
        "weekly: %s rows, seasons %s-%s%s",
        f"{len(df):,}", df["season"].min(), df["season"].max(),
        f" (unavailable: {missing})" if missing else "",
    )
    return df.reset_index(drop=True)


def load_injuries(years: list[int]) -> pd.DataFrame:
    """Per-player, per-week injury report rows.

    A season with no release yet is skipped, not fatal -- asking for the current
    season before week 1 is normal.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in years:
        path = CACHE_DIR / f"injuries_{year}.parquet"
        if not path.exists():
            try:
                path.write_bytes(_download(INJURIES_URL.format(year=year)))
            except urllib.error.HTTPError as exc:
                log.warning("no injury release for %s (%s) -- skipping", year, exc.code)
                continue
        frames.append(pd.read_parquet(path))

    if not frames:
        return pd.DataFrame(columns=INJURY_COLUMNS)

    df = pd.concat(frames, ignore_index=True)
    # `game_type`, not `season_type`: the latter only exists in newer seasons, so
    # filtering on it silently drops every older row. Same values, every season.
    df = df[df["game_type"] == "REG"]
    df["team"] = df["team"].replace(TEAM_ALIASES)
    # Releases disagree on int width across seasons; joins are on (season, week, team).
    df["season"] = df["season"].astype("int64")
    df["week"] = df["week"].astype("int64")
    return df.reset_index(drop=True)


def _load_optional(url: str, cache_name: str, years: list[int], label: str) -> pd.DataFrame:
    """Per-season parquet that may not exist yet, concatenated.

    Used for the in-season feeds. A season with no release is skipped, so a
    range that runs past today is normal rather than fatal.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in years:
        path = CACHE_DIR / f"{cache_name}_{year}.parquet"
        if not path.exists():
            try:
                path.write_bytes(_download(url.format(year=year)))
            except urllib.error.HTTPError as exc:
                log.warning("no %s release for %s (%s) -- skipping", label, year, exc.code)
                continue
        frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_depth_charts(years: list[int]) -> pd.DataFrame:
    """Team depth charts, re-snapshotted through the offseason and season."""
    return _load_optional(DEPTH_CHARTS_URL, "depth_charts", years, "depth chart")


def load_rosters(years: list[int]) -> pd.DataFrame:
    """Weekly rosters, carrying active/reserve/cut status per player."""
    return _load_optional(ROSTERS_URL, "rosters", years, "roster")


def load_schedules(years: list[int]) -> pd.DataFrame:
    """Game-level schedule, used for home/away and rest days."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / "games.csv"
    if not path.exists():
        path.write_bytes(_download(GAMES_URL))

    df = pd.read_csv(path, low_memory=False)
    df = df[(df["game_type"] == "REG") & (df["season"].isin(years))].copy()
    for col in ("home_team", "away_team"):
        df[col] = df[col].replace(TEAM_ALIASES)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    years = list(range(2016, 2026))
    w = load_weekly(years)
    s = load_schedules(years)
    i = load_injuries(years)
    print(f"weekly:    {len(w):>7,} rows  {w['season'].min()}-{w['season'].max()}")
    print(f"schedules: {len(s):>7,} rows")
    print(f"injuries:  {len(i):>7,} rows")
