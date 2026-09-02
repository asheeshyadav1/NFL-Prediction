"""Wire contract shared by the gateway and the model service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ProjectRequest(BaseModel):
    player: str = Field(..., description="Player display name (substring match)")
    season: int | None = None
    week: int | None = None


class Projection(BaseModel):
    player_id: str
    name: str
    position: str
    team: str
    opponent: str
    season: int
    week: int
    projection: float
    baseline: float = Field(..., description="Naive last-3-game average, for comparison")
    actual: float | None = Field(None, description="Set only for completed games")


class PlayerCard(Projection):
    """A projection plus enough history to judge it.

    Used by the compare view, where there is no week in the request: the
    projection is for the player's own most recent game, so `season`/`week`
    say when that was rather than being chosen by the caller.
    """

    career_games: int = Field(..., description="Projectable games on record")
    career_avg: float = Field(..., description="Mean PPR across those games")
    last6_avg: float = Field(..., description="Mean PPR over the last six")


class CompareRequest(BaseModel):
    player_a: str
    player_b: str


class Comparison(BaseModel):
    players: list[PlayerCard]
    start: str
    margin: float
    confidence: Literal["high", "moderate", "low"]
    snippets: list[dict]
    narration: str
    narration_model: str
    narration_grounded: bool


class RecommendRequest(BaseModel):
    player_a: str
    player_b: str
    season: int | None = None
    week: int | None = None


class Recommendation(BaseModel):
    players: list[Projection]
    start: str
    margin: float
    confidence: Literal["high", "moderate", "low"]
    snippets: list[dict]
    narration: str
    narration_model: str
    narration_grounded: bool
