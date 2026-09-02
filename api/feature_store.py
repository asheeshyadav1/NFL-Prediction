"""Read-side feature store.

Reuses `model/` directly rather than reimplementing feature logic -- a second
implementation is how training/serving skew gets in. Reads the cached parquet;
`scripts/load_features.py` populates the Postgres path.
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
LAST_SEASON = int(os.environ.get("LAST_SEASON", "2025"))


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
        """Every week the UI can offer, playable or not.

        Weeks with box scores are projectable. Weeks that are only scheduled
        come back with `playable: False` and the date they open, so a season
        that has not kicked off reads as upcoming rather than missing.
        """
        # Cast the groupby keys back to int: pandas widens them to float, and a
        # "Week 4.0" reaches the UI as soon as anything renders one.
        played = (
            self._frame.groupby(["season", "week"], as_index=False)
            .size()
            .rename(columns={"size": "player_weeks"})
            .sort_values(["season", "week"])
            .astype({"season": int, "week": int, "player_weeks": int})
            .to_dict("records")
        )
        for row in played:
            row["playable"] = True
            row["opens"] = None
        return played + self._upcoming()

    def _upcoming(self) -> list[dict]:
        """Scheduled weeks that have no box scores yet.

        Only ever forward of the latest played week -- an older week missing
        from the frame means no player qualified, not that it is still to come.
        """
        try:
            schedule = load_schedules([self.latest_season, self.latest_season + 1])
        except Exception as exc:  # no schedule feed is not worth failing over
            log.warning("could not read the schedule for upcoming weeks (%s)", exc)
            return []
        if schedule.empty:
            return []

        first_kickoff = (
            schedule.groupby(["season", "week"], as_index=False)["gameday"]
            .min()
            .sort_values(["season", "week"])
        )
        cutoff = (self.latest_season, self.latest_week)
        return [
            {
                "season": int(r.season),
                "week": int(r.week),
                "player_weeks": 0,
                "playable": False,
                "opens": str(r.gameday),
            }
            for r in first_kickoff.itertuples()
            if (int(r.season), int(r.week)) > cutoff
        ]

    def players(self, season: int, week: int, position: str | None = None) -> pd.DataFrame:
        m = (self._frame["season"] == season) & (self._frame["week"] == week)
        if position:
            m &= self._frame["position"] == position.upper()
        cols = ["player_id", "player_display_name", "position", "recent_team", "opponent_team"]
        return self._frame.loc[m, cols].sort_values("player_display_name")

    def career(self, player_id: str) -> dict:
        """Shape of a player's whole record, for the player-vs-player view.

        Computed over projectable rows only -- a player's first games are not
        in the frame, because the model needs history before it can speak.
        """
        rows = self._frame[self._frame["player_id"] == player_id]
        if rows.empty:
            return {"career_games": 0, "career_avg": 0.0, "last6_avg": 0.0}
        pts = rows.sort_values(["season", "week"])["fantasy_points_ppr"]
        return {
            "career_games": int(len(pts)),
            "career_avg": round(float(pts.mean()), 1),
            "last6_avg": round(float(pts.tail(6).mean()), 1),
        }

    def all_players(self) -> list[dict]:
        """Every player who has ever been projectable, newest team first.

        One row per player rather than per player-week, so the compare view can
        offer a name without the caller first picking a week it appears in.
        """
        f = self._frame.sort_values(["season", "week"])
        latest = f.groupby("player_id").tail(1)
        stats = (
            f.groupby("player_id")["fantasy_points_ppr"]
            .agg(career_games="size", career_avg="mean")
            .reset_index()
        )
        merged = latest.merge(stats, on="player_id", how="left")
        return [
            {
                "player_id": r["player_id"],
                "name": r["player_display_name"],
                "position": r["position"],
                "team": r["recent_team"],
                "last_season": int(r["season"]),
                "last_week": int(r["week"]),
                "career_games": int(r["career_games"]),
                "career_avg": round(float(r["career_avg"]), 1),
            }
            for _, r in merged.sort_values("player_display_name").iterrows()
        ]

    def latest_player_week(self, query: str) -> PlayerWeek | None:
        """The most recent game a player can be projected for.

        The compare view has no week to work from, so it uses each player's own
        last game: the model's read of their latest form, and one whose real
        result is already known.
        """
        m = self._frame["player_display_name"].str.contains(query, case=False, na=False)
        rows = self._frame[m]
        if rows.empty:
            return None
        last = rows.sort_values(["season", "week"]).iloc[-1]
        return self.lookup(last["player_id"], int(last["season"]), int(last["week"]))

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
