"""Model service -- its own deployment.

    uvicorn model_service.main:app --port 8001

This process owns the weights and the feature store and does exactly one thing:
turn a player-week into a projected point total. It has no LLM client, no
retrieval, and no knowledge that either exists.

It is split from the gateway because it scales on a different axis: it holds
model weights in memory, wants more RAM, and its load tracks projection volume
rather than web traffic. In Kubernetes it gets its own Deployment with its own
resource requests and replica count.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.feature_store import FeatureStore
from api.inference import Projector
from api.schemas import ProjectRequest, Projection

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("model-service")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Weights and features load once; a request is then just a forward pass.
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
