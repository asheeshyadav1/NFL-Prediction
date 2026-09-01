"""FastAPI gateway: `uvicorn api.main:app --reload`.

POST /project returns the model's number alone; POST /recommend adds retrieval
and narration on top of already-decided projections.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware

from api import llm, rag
from api.projection_client import build_client
from api.schemas import (
    ProjectRequest,
    Projection,
    Recommendation,
    RecommendRequest,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["projections"] = build_client()
    state["rag"] = rag.build_store()
    yield
    state.clear()


app = FastAPI(
    title="Fantasy Football Toolkit",
    description="A trained model makes the projection; the LLM only explains it.",
    lifespan=lifespan,
)
# Only used when a browser calls the API cross-origin; behind the ingress
# everything is same-origin.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _confidence(margin: float) -> Literal["high", "moderate", "low"]:
    return "high" if margin >= 3 else "moderate" if margin >= 1 else "low"


@app.get("/health")
def health(response: Response) -> dict:
    """Readiness: 503 while the model service is unreachable, so Kubernetes
    stops routing to a pod that can only return errors."""
    model = state["projections"].health()
    healthy = model.get("status") != "unreachable"
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ok" if healthy else "degraded",
        "projection_backend": state["projections"].mode,
        "retrieval_backend": state["rag"].backend,
        "model": model,
    }


@app.post("/project", response_model=Projection)
def project(req: ProjectRequest) -> Projection:
    """The model's number. No retrieval, no LLM."""
    return state["projections"].project(req.player, req.season, req.week)


@app.post("/recommend", response_model=Recommendation)
def recommend(req: RecommendRequest) -> Recommendation:
    client = state["projections"]
    a = client.project(req.player_a, req.season, req.week)
    b = client.project(req.player_b, req.season, req.week)
    if a.player_id == b.player_id:
        raise HTTPException(400, "pick two different players")

    # The decision is made here, from the projections alone.
    hi, lo = (a, b) if a.projection >= b.projection else (b, a)
    margin = round(hi.projection - lo.projection, 1)

    # One query per player: a merged query dilutes both names and matches neither.
    snippets: list[rag.Snippet] = []
    seen: set[str] = set()
    for player in (a, b):
        for snippet in state["rag"].search(
            f"{player.name} {player.team} injury status outlook", k=2
        ):
            if snippet.id not in seen:
                seen.add(snippet.id)
                snippets.append(snippet)
    snippets.sort(key=lambda s: -s.score)

    narration = llm.narrate(a.model_dump(), b.model_dump(), [s.cite() for s in snippets])

    return Recommendation(
        players=[a, b],
        start=hi.name,
        margin=margin,
        confidence=_confidence(margin),
        snippets=[
            {"player": s.player, "published": s.published, "source": s.source,
             "text": s.text, "score": round(s.score, 3)}
            for s in snippets
        ],
        narration=narration.text,
        narration_model=narration.model,
        narration_grounded=narration.grounded,
    )
