"""How the gateway talks to the model service.

HTTP when MODEL_SERVICE_URL is set, in-process otherwise. Gateway code is
identical either way.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

import httpx
from fastapi import HTTPException

from api.schemas import PlayerCard, Projection

log = logging.getLogger(__name__)


class ProjectionClient(Protocol):
    mode: str

    def project(self, player: str, season: int | None, week: int | None) -> Projection: ...
    def weeks(self) -> list[dict]: ...
    def players(self, season: int | None, week: int | None) -> list[dict]: ...
    def all_players(self) -> list[dict]: ...
    def project_latest(self, player: str) -> PlayerCard: ...
    def health(self) -> dict: ...


class InProcessClient:
    """Loads the model into this process. Used for local development."""

    mode = "in-process"

    def __init__(self) -> None:
        try:
            from api.feature_store import FeatureStore
            from api.inference import Projector
        except ImportError as exc:  # the model code is absent from the API image
            raise RuntimeError(
                f"cannot load the model in-process (missing {exc.name!r}) -- the "
                "API image ships without torch on purpose. Set MODEL_SERVICE_URL."
            ) from exc

        self._projector = Projector()
        self._store = FeatureStore()

    def project(self, player: str, season: int | None, week: int | None) -> Projection:
        season = season or self._store.latest_season
        week = week or self._store.latest_week
        pw = self._store.find_player(player, season, week)
        if pw is None:
            raise HTTPException(
                404, f"no projectable player matching {player!r} in {season} week {week}"
            )
        return Projection(
            player_id=pw.player_id, name=pw.name, position=pw.position, team=pw.team,
            opponent=pw.opponent, season=pw.season, week=pw.week,
            projection=round(self._projector.project(pw.seq, pw.ctx), 1),
            baseline=round(pw.baseline, 1),
            actual=None if pw.actual is None else round(pw.actual, 1),
        )

    def weeks(self) -> list[dict]:
        return self._store.weeks()

    def all_players(self) -> list[dict]:
        return self._store.all_players()

    def project_latest(self, player: str) -> PlayerCard:
        pw = self._store.latest_player_week(player)
        if pw is None:
            raise HTTPException(404, f"no projectable player matching {player!r}")
        return PlayerCard(
            player_id=pw.player_id, name=pw.name, position=pw.position, team=pw.team,
            opponent=pw.opponent, season=pw.season, week=pw.week,
            projection=round(self._projector.project(pw.seq, pw.ctx), 1),
            baseline=round(pw.baseline, 1),
            actual=None if pw.actual is None else round(pw.actual, 1),
            **self._store.career(pw.player_id),
        )

    def players(self, season: int | None, week: int | None) -> list[dict]:
        df = self._store.players(
            season or self._store.latest_season, week or self._store.latest_week
        )
        return df.rename(
            columns={
                "player_display_name": "name",
                "recent_team": "team",
                "opponent_team": "opponent",
            }
        ).to_dict("records")

    def health(self) -> dict:
        return {
            "val_mae": round(self._projector.val_mae, 3),
            "latest_season": self._store.latest_season,
            "latest_week": self._store.latest_week,
        }


class HttpClient:
    """Calls the model service over the network. Used in Kubernetes."""

    mode = "http"

    def __init__(self, base_url: str, timeout: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def project(self, player: str, season: int | None, week: int | None) -> Projection:
        payload = {"player": player, "season": season, "week": week}
        try:
            response = self._client.post("/project", json=payload)
        except httpx.HTTPError as exc:
            raise HTTPException(503, f"model service unreachable: {exc}") from exc
        if response.status_code == 404:
            # Pass the model service's 404 through, not as a gateway failure.
            raise HTTPException(404, response.json().get("detail", "player not found"))
        response.raise_for_status()
        return Projection(**response.json())

    def weeks(self) -> list[dict]:
        return self._get("/weeks")

    def all_players(self) -> list[dict]:
        return self._get("/players/all")

    def project_latest(self, player: str) -> PlayerCard:
        try:
            response = self._client.post("/project/latest", json={"player": player})
        except httpx.HTTPError as exc:
            raise HTTPException(503, f"model service unreachable: {exc}") from exc
        if response.status_code == 404:
            raise HTTPException(404, response.json().get("detail", "player not found"))
        response.raise_for_status()
        return PlayerCard(**response.json())

    def players(self, season: int | None, week: int | None) -> list[dict]:
        params = {k: v for k, v in {"season": season, "week": week}.items() if v}
        return self._get("/players", params=params)

    def _get(self, path: str, **kwargs) -> list[dict]:
        try:
            response = self._client.get(path, **kwargs)
        except httpx.HTTPError as exc:
            raise HTTPException(503, f"model service unreachable: {exc}") from exc
        response.raise_for_status()
        return response.json()

    def health(self) -> dict:
        try:
            return self._client.get("/health").json()
        except httpx.HTTPError as exc:
            return {"status": "unreachable", "error": str(exc)}


def build_client() -> ProjectionClient:
    url = os.environ.get("MODEL_SERVICE_URL")
    if url:
        log.info("projection backend: model service at %s", url)
        return HttpClient(url)
    log.info("projection backend: in-process (set MODEL_SERVICE_URL to split them)")
    return InProcessClient()
