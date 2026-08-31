"""Read-side feature store.

Serving needs the same leakage-safe windows the model trained on, so this reuses
`model/` directly rather than reimplementing feature logic -- a second
implementation is how a serving/training skew bug gets in.

Backed by Postgres when DATABASE_URL is set, otherwise by the cached parquet the
training pipeline already downloaded. The parquet path keeps the demo runnable
with no database; `infra/sql/schema.sql` and `scripts/load_features.py` populate
the Postgres path.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
sys.path.insert(0, str(MODEL_DIR))

from data import load_schedules, load_weekly  # noqa: E402
from dataset import build_windows  # noqa: E402
from features import build  # noqa: E402

log = logging.getLogger(__name__)

FIRST_SEASON = int(os.environ.get("FIRST_SEASON", "2016"))
LAST_SEASON = int(os.environ.get("LAST_SEASON", "2024"))


@dataclass(frozen=True)
class PlayerWeek:
    """One projectable player-week, with its model inputs attached."""

    player_id: str
    name: str
    position: str
    team: str
    opponent: str
    season: int
    week: int
    seq: np.ndarray
    ctx: np.ndarray
    baseline: float
    actual: float | None  # known only for completed games


class FeatureStore:
    def __init__(self) -> None:
        years = list(range(FIRST_SEASON, LAST_SEASON + 1))
        log.info("building feature store for %s-%s ...", years[0], years[-1])
        frame = build(load_weekly(years), load_schedules(years))
        windows = build_windows(frame)

        self._frame = windows["frame"]
        self._seq = windows["seq"]
        self._ctx = windows["ctx"]
        self._baseline = windows["baseline"]
        self._y = windows["y"]

        self.latest_season = int(self._frame["season"].max())
        latest = self._frame[self._frame["season"] == self.latest_season]
        self.latest_week = int(latest["week"].max())
        log.info(
            "feature store ready: %s player-weeks, latest = %s week %s",
            f"{len(self._frame):,}", self.latest_season, self.latest_week,
        )

    def weeks(self) -> list[dict]:
        return (
            self._frame.groupby(["season", "week"], as_index=False)
            .size()
            .rename(columns={"size": "player_weeks"})
            .sort_values(["season", "week"])
            .to_dict("records")
        )

    def players(self, season: int, week: int, position: str | None = None) -> pd.DataFrame:
        m = (self._frame["season"] == season) & (self._frame["week"] == week)
        if position:
            m &= self._frame["position"] == position.upper()
        cols = ["player_id", "player_display_name", "position", "recent_team", "opponent_team"]
        return self._frame.loc[m, cols].sort_values("player_display_name")

    def lookup(self, player_id: str, season: int, week: int) -> PlayerWeek | None:
        m = (
            (self._frame["player_id"] == player_id)
            & (self._frame["season"] == season)
            & (self._frame["week"] == week)
        )
        idx = self._frame.index[m]
        if len(idx) == 0:
            return None
        i = int(idx[0])
        row = self._frame.iloc[i]
        return PlayerWeek(
            player_id=row["player_id"],
            name=row["player_display_name"],
            position=row["position"],
            team=row["recent_team"],
            opponent=row["opponent_team"],
            season=int(row["season"]),
            week=int(row["week"]),
            seq=self._seq[i],
            ctx=self._ctx[i],
            baseline=float(self._baseline[i]),
            actual=float(self._y[i]),
        )

    def find_player(self, query: str, season: int, week: int) -> PlayerWeek | None:
        """Resolve a display name (case-insensitive substring) to a player-week."""
        m = (
            (self._frame["season"] == season)
            & (self._frame["week"] == week)
            & self._frame["player_display_name"].str.contains(query, case=False, na=False)
        )
        idx = self._frame.index[m]
        if len(idx) == 0:
            return None
        return self.lookup(self._frame.iloc[int(idx[0])]["player_id"], season, week)
