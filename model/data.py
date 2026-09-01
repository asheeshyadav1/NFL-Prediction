"""Raw data pull from nflverse, with on-disk caching.

Nothing here does feature work -- it only fetches and caches. Keeping the network
boundary in one file means the rest of the pipeline is deterministic and runnable
offline once the cache is warm.

We read the nflverse-data release files directly over HTTPS rather than going
through `nfl_data_py`, which pins `pandas<2` and cannot coexist with a current
torch/pandas stack. Same upstream files, one less dependency.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
import urllib.request

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

_RELEASE = "https://github.com/nflverse/nflverse-data/releases/download/player_stats"
# nflverse renamed these assets partway through; a few seasons (2019 at time of
# writing) only exist under the legacy name, so try both.
WEEKLY_URLS = (
    _RELEASE + "/stats_player_week_{year}.parquet",
    _RELEASE + "/player_stats_{year}.parquet",
)
GAMES_URL = "http://www.habitatring.com/games.csv"

# The official NFL injury report -- the same practice/game-status data that
# powers nfl.com/injuries, mirrored by nflverse as a season parquet. That page
# itself renders client-side off a token-gated internal API, so this release is
# both the more stable source and the one that joins to our player ids.
INJURIES_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{year}.parquet"
)

# Positions we project. Kickers and team defenses score through a completely
# different process and would need their own model.
POSITIONS = ("QB", "RB", "WR", "TE")

# Renames to a stable internal schema, so a change upstream is a one-line fix.
COLUMN_ALIASES = {
    "team": "recent_team",
    "passing_interceptions": "interceptions",
}

# The schedule feed keeps a franchise's historical abbreviation; the stats feed
# backfills the current one. Without this the relocated franchises fail to join.
TEAM_ALIASES = {"SD": "LAC", "OAK": "LV", "STL": "LA"}

# Enough of the injury schema to build an empty frame that still has the columns
# downstream code selects on.
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
    """Per-player, per-week box scores including precomputed PPR points."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in years:
        path = CACHE_DIR / f"weekly_{year}.parquet"
        if not path.exists():
            path.write_bytes(_download_first(WEEKLY_URLS, year=year))
        season = pd.read_parquet(path)
        season = season.rename(
            columns={k: v for k, v in COLUMN_ALIASES.items() if k in season.columns}
        )
        frames.append(season)

    df = pd.concat(frames, ignore_index=True)
    df = df[df["season_type"] == "REG"]
    df = df[df["position"].isin(POSITIONS)]
    return df.reset_index(drop=True)


def load_injuries(years: list[int]) -> pd.DataFrame:
    """Per-player, per-week injury report rows.

    A season is skipped rather than fatal when nflverse has not cut a release
    for it yet -- asking for the current season before week 1 is a normal thing
    to do, and it should not take the ingest down.
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
    # `game_type`, not `season_type`: the injuries release gained `season_type`
    # partway through (2025 has it, 2024 does not), so filtering on that silently
    # concats a column of NaN for the older seasons and drops every one of their
    # rows. `game_type` carries the same REG/WC/DIV/CON/SB values in every season.
    df = df[df["game_type"] == "REG"]
    df["team"] = df["team"].replace(TEAM_ALIASES)
    # The releases disagree on int width across seasons; downstream joins are on
    # (season, week, team), so normalise here rather than at each call site.
    df["season"] = df["season"].astype("int64")
    df["week"] = df["week"].astype("int64")
    return df.reset_index(drop=True)


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
    years = list(range(2016, 2025))
    w = load_weekly(years)
    s = load_schedules(years)
    i = load_injuries(years)
    print(f"weekly:    {len(w):>7,} rows  {w['season'].min()}-{w['season'].max()}")
    print(f"schedules: {len(s):>7,} rows")
    print(f"injuries:  {len(i):>7,} rows")
