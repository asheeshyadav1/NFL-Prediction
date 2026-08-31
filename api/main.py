"""FastAPI gateway.

    uvicorn api.main:app --reload

Two endpoints matter:
  POST /project    -- the model's number, and nothing else
  POST /recommend  -- projection + retrieval + narration

This is where the separation the project is built on becomes visible:
`/recommend` gets the projections first, then hands the *already-decided*
numbers to the retriever and the narrator. Neither can revise them -- the
retriever returns only text, and the narration is checked against the numbers
we handed it before it goes out.
"""

from __future__ import annotations

import logging
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _confidence(margin: float) -> Literal["high", "moderate", "low"]:
    return "high" if margin >= 3 else "moderate" if margin >= 1 else "low"


@app.get("/health")
def health(response: Response) -> dict:
    """Readiness, not liveness.

    The gateway cannot answer a single useful request without the model service,
    so an unreachable model service is reported as `degraded` with a 503 -- a
    flat `ok` here would let Kubernetes route traffic to a pod that can only
    return errors.
    """
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

    # One query per player rather than a merged one: a combined query dilutes
    # both names and reliably retrieves neither player's news.
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
