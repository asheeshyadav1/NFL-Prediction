"""Service-level tests.

Pin the contract the project rests on: the number comes from the model, and the
narration cannot change it. A stub projection client keeps these independent of
trained weights and the nflverse cache.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import llm, main, rag
from api.schemas import Projection


class StubProjections:
    """Stands in for the model service with fixed, known numbers."""

    mode = "stub"

    PLAYERS = {
        "alpha": Projection(
            player_id="P1", name="Alpha Back", position="RB", team="AAA",
            opponent="BBB", season=2024, week=18, projection=18.4,
            baseline=15.0, actual=21.0,
        ),
        "beta": Projection(
            player_id="P2", name="Beta Back", position="RB", team="CCC",
            opponent="DDD", season=2024, week=18, projection=11.2,
            baseline=12.5, actual=6.0,
        ),
    }

    def project(self, player: str, season, week) -> Projection:
        from fastapi import HTTPException

        for key, value in self.PLAYERS.items():
            if key in player.lower() or value.name.lower() == player.lower():
                return value
        raise HTTPException(404, f"no player matching {player!r}")

    def health(self) -> dict:
        return {"val_mae": 4.376}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(main, "build_client", lambda: StubProjections())
    monkeypatch.setattr(
        rag,
        "load_snippets",
        lambda *a, **k: [
            rag.Snippet(id="n1", player="Alpha Back", team="AAA",
                        published="2025-01-02", source="TEST",
                        text="Alpha Back is questionable with an ankle injury."),
            rag.Snippet(id="n2", player="Beta Back", team="CCC",
                        published="2025-01-02", source="TEST",
                        text="Beta Back has no injury designation."),
        ],
    )
    with TestClient(main.app) as c:
        yield c


def test_health_reports_both_backends(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["projection_backend"] == "stub"
    assert body["retrieval_backend"] == "in-memory"


def test_project_returns_the_model_number_only(client: TestClient) -> None:
    body = client.post("/project", json={"player": "alpha"}).json()
    assert body["projection"] == 18.4
    assert body["baseline"] == 15.0
    # /project is the bare model path -- no narration, no retrieval.
    assert "narration" not in body and "snippets" not in body


def test_unknown_player_is_a_404(client: TestClient) -> None:
    assert client.post("/project", json={"player": "nobody"}).status_code == 404


def test_recommendation_follows_the_projection(client: TestClient) -> None:
    body = client.post(
        "/recommend", json={"player_a": "beta", "player_b": "alpha"}
    ).json()
    # Alpha projects higher, so Alpha starts -- regardless of argument order.
    assert body["start"] == "Alpha Back"
    assert body["margin"] == pytest.approx(7.2, abs=0.05)
    assert body["confidence"] == "high"


def test_narration_quotes_the_model_numbers(client: TestClient) -> None:
    body = client.post(
        "/recommend", json={"player_a": "alpha", "player_b": "beta"}
    ).json()
    assert body["narration_grounded"] is True
    assert "18.4" in body["narration"] and "11.2" in body["narration"]


def test_retrieval_surfaces_both_players(client: TestClient) -> None:
    body = client.post(
        "/recommend", json={"player_a": "alpha", "player_b": "beta"}
    ).json()
    players = {s["player"] for s in body["snippets"]}
    assert players == {"Alpha Back", "Beta Back"}


def test_comparing_a_player_to_themselves_is_rejected(client: TestClient) -> None:
    res = client.post("/recommend", json={"player_a": "alpha", "player_b": "Alpha Back"})
    assert res.status_code == 400


def test_grounding_check_catches_a_drifting_narration() -> None:
    """The guard that would catch an LLM inventing its own numbers."""
    players = [{"projection": 18.4}, {"projection": 11.2}]
    assert llm._verify("Start Alpha (18.4) over Beta (11.2).", players)
    assert not llm._verify("Start Alpha (19.0) over Beta (10.0).", players)


def test_retrieval_ranks_the_named_player_first() -> None:
    store = rag.InMemoryStore(
        [
            rag.Snippet(id="a", player="Alpha Back", team="AAA", published="2025-01-01",
                        source="TEST", text="Questionable with an ankle injury."),
            rag.Snippet(id="b", player="Zeta Wideout", team="ZZZ", published="2025-01-01",
                        source="TEST", text="Questionable with an ankle injury."),
        ]
    )
    # Identical injury text -- only the player name distinguishes them, which is
    # exactly why the name is part of the embedded document.
    assert store.search("Alpha Back AAA injury status", k=1)[0].player == "Alpha Back"


class UnreachableProjections(StubProjections):
    """Model service down: the gateway can serve nothing useful."""

    def health(self) -> dict:
        return {"status": "unreachable", "error": "connection refused"}


def test_health_is_degraded_when_the_model_service_is_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "build_client", lambda: UnreachableProjections())
    monkeypatch.setattr(rag, "load_snippets", lambda *a, **k: [])
    with TestClient(main.app) as c:
        res = c.get("/health")
    # 503 so a readiness probe pulls the pod out of the load balancer rather
    # than routing traffic to a gateway that can only return errors.
    assert res.status_code == 503
    assert res.json()["status"] == "degraded"


# --- narration punctuation -------------------------------------------------
#
# The UI carries no em dashes. The system prompt asks the narrator for none,
# but a prompt is a request rather than a guarantee, so the text is normalised
# on the way out as well. These pin the normaliser, since a live model cannot
# be relied on to produce the case under test.

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Start Chase — he is healthy.", "Start Chase, he is healthy."),
        ("Start Chase – he is healthy.", "Start Chase, he is healthy."),
        ("Chase (16.7)—Jefferson (9.9)", "Chase (16.7), Jefferson (9.9)"),
        ("nothing to change here", "nothing to change here"),
    ],
)
def test_narration_carries_no_em_dashes(raw: str, expected: str) -> None:
    assert llm._no_em_dashes(raw) == expected


def test_normaliser_leaves_the_numbers_alone() -> None:
    # It runs before the grounding check, so it must not touch any digits.
    text = "Chase 16.7 PPR — Jefferson 9.9 PPR, a 6.8-point edge."
    cleaned = llm._no_em_dashes(text)
    for number in ("16.7", "9.9", "6.8"):
        assert number in cleaned
    assert "—" not in cleaned


def test_template_narration_has_no_dash_stand_ins() -> None:
    a = {"name": "A", "position": "WR", "team": "CIN", "opponent": "BAL", "projection": 16.7}
    b = {"name": "B", "position": "WR", "team": "MIN", "opponent": "DAL", "projection": 9.9}
    text = llm._template(a, b, [])
    assert "—" not in text and "–" not in text and " -- " not in text
