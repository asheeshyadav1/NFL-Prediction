"""Leakage tests.

These are the most important tests in the repo. The model's headline number is
only meaningful if no future information reaches the features, and leakage is
silent -- it makes the metrics *better*, so nothing else will catch it. Each
test below pins one of the guarantees the pipeline claims.

They run on a small synthetic frame so they're fast and deterministic; the
invariants they check are structural, not data-dependent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
sys.path.insert(0, str(MODEL_DIR))

from dataset import MIN_HISTORY, SEQ_LEN, build_windows, temporal_split  # noqa: E402
from features import SEQ_STATS, build  # noqa: E402


def _synthetic(n_players: int = 6, n_weeks: int = 17) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A tiny league: deterministic, with each player-week uniquely identifiable."""
    teams = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    rows = []
    for season in (2021, 2022, 2023):
        for week in range(1, n_weeks + 1):
            for p in range(n_players):
                team = teams[p % len(teams)]
                opp = teams[(p + 1) % len(teams)]
                rows.append(
                    {
                        "player_id": f"P{p:02d}",
                        "player_display_name": f"Player {p}",
                        "position": ["QB", "RB", "WR", "TE"][p % 4],
                        "season": season,
                        "week": week,
                        "season_type": "REG",
                        "recent_team": team,
                        "opponent_team": opp,
                        # Unique per player-week so leakage is detectable by value.
                        "fantasy_points_ppr": float(season * 1000 + week * 10 + p),
                        **{c: float(week + p) for c in SEQ_STATS if c != "fantasy_points_ppr"},
                    }
                )
    weekly = pd.DataFrame(rows)

    games = []
    for season in (2021, 2022, 2023):
        for week in range(1, n_weeks + 1):
            for i in range(0, len(teams), 2):
                games.append(
                    {
                        "season": season, "week": week, "game_type": "REG",
                        "gameday": f"{season}-09-{(week % 28) + 1:02d}",
                        "home_team": teams[i], "away_team": teams[i + 1],
                    }
                )
    return weekly, pd.DataFrame(games)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    weekly, schedules = _synthetic()
    return build(weekly, schedules)


def test_target_never_appears_in_its_own_window(frame: pd.DataFrame) -> None:
    """The single most important invariant: week t's box score is not an input."""
    w = build_windows(frame)
    ppr_index = SEQ_STATS.index("fantasy_points_ppr")
    ppr_history = w["seq"][:, :, ppr_index]

    for i, target in enumerate(w["y"]):
        assert target not in set(ppr_history[i].tolist()), (
            f"row {i}: the target ({target}) appears inside its own input window"
        )


def test_window_contains_only_strictly_prior_games(frame: pd.DataFrame) -> None:
    """Every value in a window comes from an earlier week for the same player."""
    w = build_windows(frame)
    kept = w["frame"].reset_index(drop=True)
    ppr_index = SEQ_STATS.index("fantasy_points_ppr")

    ordered = frame.sort_values(["player_id", "season", "week"])
    by_player = {
        pid: grp["fantasy_points_ppr"].tolist() for pid, grp in ordered.groupby("player_id")
    }
    order = {
        pid: list(zip(grp["season"], grp["week"])) for pid, grp in ordered.groupby("player_id")
    }

    for i in range(len(kept)):
        pid = kept.loc[i, "player_id"]
        pos = order[pid].index((kept.loc[i, "season"], kept.loc[i, "week"]))
        prior = set(by_player[pid][:pos])
        window = [v for v in w["seq"][i, :, ppr_index] if v != 0.0]  # 0.0 = left padding
        assert set(window) <= prior, f"row {i}: window contains non-prior values"


def test_rolling_features_exclude_the_current_week(frame: pd.DataFrame) -> None:
    """roll3_ppr is the mean of the 3 prior games -- never including this one."""
    ordered = frame.sort_values(["player_id", "season", "week"])
    for _, grp in ordered.groupby("player_id"):
        ppr = grp["fantasy_points_ppr"].to_numpy()
        roll3 = grp["roll3_ppr"].to_numpy()
        for i in range(MIN_HISTORY, len(grp)):
            expected = ppr[max(0, i - 3): i].mean()
            assert roll3[i] == pytest.approx(expected), (
                f"roll3_ppr at position {i} used the current week"
            )


def test_opponent_strength_excludes_the_current_week(frame: pd.DataFrame) -> None:
    """A defense's rating for week N must not contain week N's own result."""
    ordered = frame.sort_values(["season", "week"])
    grouped = (
        ordered.groupby(["season", "week", "opponent_team", "position"], as_index=False)
        .agg(allowed=("fantasy_points_ppr", "sum"),
             rating=("opp_def_allowed_prior", "first"))
        .sort_values(["opponent_team", "position", "season", "week"])
    )
    for (_, _), grp in grouped.groupby(["opponent_team", "position"]):
        allowed = grp["allowed"].to_numpy()
        rating = grp["rating"].to_numpy()
        for i in range(1, len(grp)):
            expected = allowed[:i].mean()
            assert rating[i] == pytest.approx(expected), (
                "opponent rating includes the week it is being used to predict"
            )


def test_baseline_is_the_documented_naive_rule(frame: pd.DataFrame) -> None:
    """The baseline the model is compared against really is 'last 3 games'."""
    w = build_windows(frame)
    assert np.allclose(w["baseline"], w["frame"]["roll3_ppr"].to_numpy())


def test_split_is_temporal_not_random(frame: pd.DataFrame) -> None:
    split = temporal_split(frame, val_season=2022, test_season=2023)
    assert split.train["season"].max() < split.val["season"].min()
    assert split.val["season"].max() < split.test["season"].min()
    # And no player-week may appear in more than one split.
    def keys(d: pd.DataFrame) -> set:
        return set(zip(d["player_id"], d["season"], d["week"]))

    assert not keys(split.train) & keys(split.val)
    assert not keys(split.val) & keys(split.test)
    assert not keys(split.train) & keys(split.test)


def test_rows_without_enough_history_are_dropped(frame: pd.DataFrame) -> None:
    """A projection needs history; so does the baseline it's compared against."""
    w = build_windows(frame)
    assert (w["frame"]["games_played"] >= MIN_HISTORY).all()
    assert w["seq"].shape[1] == SEQ_LEN
    assert not np.isnan(w["baseline"]).any()


# --- retrieval is scoped to the week being decided -------------------------
#
# The corpus spans every season the app can serve, so an unscoped query can
# hand the narrator a report filed after the game it is explaining. That is
# hindsight reaching the user as "context", which is the same class of error
# the window tests above exist to prevent.

def _store():
    from api import rag

    return rag.InMemoryStore(
        [
            rag.Snippet(id="a", player="Davante Adams", team="GB", published="2019-10-20",
                        season=2019, week=7, source="NFL-INJURY-REPORT",
                        text="Davante Adams is listed as Out with a toe injury."),
            rag.Snippet(id="b", player="Davante Adams", team="GB", published="2019-12-15",
                        season=2019, week=15, source="NFL-INJURY-REPORT",
                        text="Davante Adams is listed as Questionable with a toe injury."),
            rag.Snippet(id="c", player="Davante Adams", team="LV", published="2024-10-20",
                        season=2024, week=7, source="NFL-INJURY-REPORT",
                        text="Davante Adams is listed as Out with a hamstring injury."),
            rag.Snippet(id="d", player="Davante Adams", team="GB", published="2019-01-01",
                        source="SYNTHETIC-DEMO", text="Davante Adams demo fixture line."),
        ]
    )


def test_retrieval_never_cites_a_later_week() -> None:
    hits = _store().search("Davante Adams GB injury status outlook", k=4, season=2019, week=8)
    weeks = {h.week for h in hits if h.season is not None}
    assert weeks == {7}, f"expected only week <= 8 of 2019, got {weeks}"


def test_retrieval_never_cites_another_season() -> None:
    hits = _store().search("Davante Adams injury status outlook", k=4, season=2019, week=18)
    seasons = {h.season for h in hits if h.season is not None}
    assert seasons == {2019}, f"cross-season citation leaked: {seasons}"


def test_undated_fixture_stays_eligible() -> None:
    hits = _store().search("Davante Adams GB injury status outlook", k=4, season=2019, week=8)
    assert "d" in {h.id for h in hits}, "undated demo fixture should remain retrievable"


def test_unscoped_search_still_returns_everything() -> None:
    hits = _store().search("Davante Adams injury status outlook", k=4)
    assert len(hits) == 4, "an unscoped query must keep its previous behaviour"
