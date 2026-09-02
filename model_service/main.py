"""Model service: `uvicorn model_service.main:app --port 8001`.

Owns the weights and feature store and does one thing -- player-week to
projected points. No LLM, no retrieval. Its own deployment because it scales on
projection volume rather than web traffic.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.feature_store import FeatureStore
from api.inference import Projector
from api.schemas import PlayerCard, ProjectRequest, Projection

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("model-service")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loaded once; a request is then just a forward pass.
    state["projector"] = Projector()
    state["store"] = FeatureStore()
    yield
    state.clear()


app = FastAPI(title="Fantasy Football Toolkit model service", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    store: FeatureStore = state["store"]
    return {
        "status": "ok",
        "service": "model",
        "val_mae": round(state["projector"].val_mae, 3),
        "latest_season": store.latest_season,
        "latest_week": store.latest_week,
    }


@app.get("/weeks")
def weeks() -> list[dict]:
    return state["store"].weeks()


@app.get("/players")
def players(season: int | None = None, week: int | None = None,
            position: str | None = None) -> list[dict]:
    store: FeatureStore = state["store"]
    df = store.players(season or store.latest_season, week or store.latest_week, position)
    return df.rename(
        columns={
            "player_display_name": "name",
            "recent_team": "team",
            "opponent_team": "opponent",
        }
    ).to_dict("records")


@app.get("/players/all")
def all_players() -> list[dict]:
    return state["store"].all_players()


@app.post("/project/latest", response_model=PlayerCard)
def project_latest(req: dict) -> PlayerCard:
    store: FeatureStore = state["store"]
    pw = store.latest_player_week(req["player"])
    if pw is None:
        raise HTTPException(404, f"no projectable player matching {req['player']!r}")
    return PlayerCard(
        player_id=pw.player_id, name=pw.name, position=pw.position, team=pw.team,
        opponent=pw.opponent, season=pw.season, week=pw.week,
        projection=round(state["projector"].project(pw.seq, pw.ctx), 1),
        baseline=round(pw.baseline, 1),
        actual=None if pw.actual is None else round(pw.actual, 1),
        **store.career(pw.player_id),
    )


@app.post("/project", response_model=Projection)
def project(req: ProjectRequest) -> Projection:
    store: FeatureStore = state["store"]
    season = req.season or store.latest_season
    week = req.week or store.latest_week

    pw = store.find_player(req.player, season, week)
    if pw is None:
        raise HTTPException(
            404, f"no projectable player matching {req.player!r} in {season} week {week}"
        )
    projection = state["projector"].project(pw.seq, pw.ctx)
    return Projection(
        player_id=pw.player_id, name=pw.name, position=pw.position, team=pw.team,
        opponent=pw.opponent, season=pw.season, week=pw.week,
        projection=round(projection, 1), baseline=round(pw.baseline, 1),
        actual=None if pw.actual is None else round(pw.actual, 1),
    )
