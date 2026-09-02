"""Feature engineering.

One rule: every feature on a player-week row must be computable before that game
kicks off. Rolling stats are shifted a game, opponent strength is an expanding
mean over strictly prior weeks. If you add a feature here, shift it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The sequence fed to the LSTM: one vector per historical game.
SEQ_STATS = [
    "fantasy_points_ppr",
    "targets",
    "receptions",
    "receiving_yards",
    "carries",
    "rushing_yards",
    "passing_yards",
    "passing_tds",
    "rushing_tds",
    "receiving_tds",
    "interceptions",
    "fumbles_lost",
    "target_share",
]

# Lost fumbles are spread across three columns upstream and score -2 each. They
# are already inside the PPR target, so without them the model is asked to
# predict a penalty it cannot see coming.
FUMBLE_COLUMNS = ("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost")

# Context known ahead of kickoff for the week being projected.
CONTEXT_FEATURES = [
    "week",
    "is_home",
    "rest_days",
    "games_played",
    "roll3_ppr",
    "roll5_ppr",
    "roll3_targets",
    "roll3_carries",
    "roll5_target_share",
    "opp_def_allowed_prior",
    "is_QB",
    "is_RB",
    "is_WR",
    "is_TE",
]

TARGET = "fantasy_points_ppr"


def _team_week_context(schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team) with home flag and days of rest."""
    home = schedules[["season", "week", "gameday", "home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opp"}
    )
    home["is_home"] = 1
    away = schedules[["season", "week", "gameday", "away_team", "home_team"]].rename(
        columns={"away_team": "team", "home_team": "opp"}
    )
    away["is_home"] = 0

    tw = pd.concat([home, away], ignore_index=True)
    tw["gameday"] = pd.to_datetime(tw["gameday"])
    tw = tw.sort_values(["team", "season", "week"])

    prev = tw.groupby(["team", "season"])["gameday"].shift(1)
    tw["rest_days"] = (tw["gameday"] - prev).dt.days
    # Week 1 (and any gap we cannot measure) gets a normal week of rest.
    tw["rest_days"] = tw["rest_days"].fillna(7).clip(3, 21)

    return tw[["season", "week", "team", "is_home", "rest_days"]]


def _opponent_strength(weekly: pd.DataFrame) -> pd.DataFrame:
    """PPR points a defense has allowed to a position, prior weeks only.

    An expanding mean shifted one game, so a rating for week N never contains
    week N's own result.
    """
    allowed = (
        weekly.groupby(["season", "week", "opponent_team", "position"], as_index=False)[
            "fantasy_points_ppr"
        ]
        .sum()
        .rename(columns={"opponent_team": "def_team", "fantasy_points_ppr": "allowed"})
        .sort_values(["def_team", "position", "season", "week"])
    )

    grp = allowed.groupby(["def_team", "position"])["allowed"]
    allowed["opp_def_allowed_prior"] = grp.transform(
        lambda s: s.shift(1).expanding().mean()
    )

    # Early rows have no history; fall back to the position's prior-only mean.
    pos_mean = (
        allowed.sort_values(["position", "season", "week"])
        .groupby("position")["allowed"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    allowed["opp_def_allowed_prior"] = allowed["opp_def_allowed_prior"].fillna(pos_mean)

    return allowed[["season", "week", "def_team", "position", "opp_def_allowed_prior"]]


def build(weekly: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Return a modelling frame: one row per projectable player-week."""
    df = weekly.copy()

    # Derived before the SEQ_STATS check below, which is what makes it required.
    df["fumbles_lost"] = sum(
        pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        for c in FUMBLE_COLUMNS
        if c in df.columns
    )

    for col in SEQ_STATS:
        if col not in df.columns:
            raise KeyError(f"expected column missing from weekly data: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    # --- rolling player form (shifted: prior games only) ---
    g = df.groupby("player_id")
    df["games_played"] = g.cumcount()
    for col, windows in (("fantasy_points_ppr", (3, 5)), ("targets", (3,)), ("carries", (3,)), ("target_share", (5,))):
        for w in windows:
            name = f"roll{w}_{'ppr' if col == 'fantasy_points_ppr' else col}"
            df[name] = g[col].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())

    # --- schedule context ---
    tw = _team_week_context(schedules)
    df = df.merge(
        tw, left_on=["season", "week", "recent_team"], right_on=["season", "week", "team"], how="left"
    ).drop(columns=["team"])
    unmatched = df["is_home"].isna().mean()
    if unmatched > 0.02:
        raise RuntimeError(
            f"{unmatched:.1%} of player-weeks failed to join the schedule -- "
            "likely a team-abbreviation mismatch. Fix the join rather than filling."
        )
    df["is_home"] = df["is_home"].fillna(0.0)
    df["rest_days"] = df["rest_days"].fillna(7.0)

    # --- opponent strength ---
    strength = _opponent_strength(df)
    df = df.merge(
        strength,
        left_on=["season", "week", "opponent_team", "position"],
        right_on=["season", "week", "def_team", "position"],
        how="left",
    ).drop(columns=["def_team"])
    df["opp_def_allowed_prior"] = df["opp_def_allowed_prior"].fillna(
        df.groupby("position")["opp_def_allowed_prior"].transform("mean")
    )

    # --- position one-hots ---
    for pos in ("QB", "RB", "WR", "TE"):
        df[f"is_{pos}"] = (df["position"] == pos).astype(float)

    df[CONTEXT_FEATURES] = df[CONTEXT_FEATURES].astype(float).fillna(0.0)
    return df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def naive_baseline(df: pd.DataFrame) -> np.ndarray:
    """'Project = last 3 games' -- the number the model has to beat."""
    return df["roll3_ppr"].to_numpy(dtype=np.float32)
