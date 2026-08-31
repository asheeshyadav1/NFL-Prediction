"""Schemas shared by the gateway and the model service.

Both services import these so the wire contract between them is defined once.
"""

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
