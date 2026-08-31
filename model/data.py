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
    print(f"weekly:    {len(w):>7,} rows  {w['season'].min()}-{w['season'].max()}")
    print(f"schedules: {len(s):>7,} rows")
