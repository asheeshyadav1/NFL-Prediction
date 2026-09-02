"""Injury-ingest tests.

The ingest is the only place real third-party data enters the retrieval corpus,
and the failure that matters is silent: a schema change upstream that drops rows
without erroring. These pin the transform, not the network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from api import rag  # noqa: E402
from ingest_injuries import SOURCE, build_snippets, describe  # noqa: E402


def _schedule() -> pd.DataFrame:
    return pd.DataFrame([
        {"season": 2025, "week": 3, "home_team": "BAL", "away_team": "CIN",
         "gameday": "2025-09-21"},
    ])


def _report(**overrides) -> dict:
    row = {
        "season": 2025, "week": 3, "team": "BAL", "gsis_id": "00-0011111",
        "position": "TE", "full_name": "Test Player",
        "report_primary_injury": "Knee", "report_secondary_injury": None,
        "report_status": "Questionable",
        "practice_primary_injury": "Knee", "practice_secondary_injury": None,
        "practice_status": "Limited Participation in Practice",
    }
    row.update(overrides)
    return row


def test_snippet_carries_real_source_and_game_date():
    [snippet] = build_snippets(pd.DataFrame([_report()]), _schedule())

    assert snippet["source"] == SOURCE != "SYNTHETIC-DEMO"
    assert snippet["id"] == "inj-2025-03-00-0011111"
    # Dated by the game it applies to, not by ingest time.
    assert snippet["published"] == "2025-09-21"
    assert "Questionable" in snippet["text"] and "knee" in snippet["text"]


def test_non_skill_positions_are_dropped():
    rows = pd.DataFrame([_report(position="G"), _report(position="WR",
                                                        gsis_id="00-0022222")])
    assert [s["player"] for s in build_snippets(rows, _schedule())] == ["Test Player"]


def test_rows_with_no_status_at_all_are_dropped():
    rows = pd.DataFrame([_report(report_status=None, practice_status=None)])
    assert build_snippets(rows, _schedule()) == []


def test_practice_only_row_survives():
    """Mid-week there is no game status yet -- practice is the only signal."""
    [snippet] = build_snippets(
        pd.DataFrame([_report(report_status=None, report_primary_injury=None)]),
        _schedule(),
    )
    assert "no game-status designation" in snippet["text"]
    assert "limited participation" in snippet["text"]


def test_rested_player_is_not_described_as_injured():
    text = describe(pd.Series(_report(
        report_status=None, report_primary_injury=None,
        practice_primary_injury="Not injury related - resting player",
        practice_status="Did Not Participate In Practice",
    )))
    assert "rested, not injured" in text
    assert "not injury related" not in text.lower().replace("rested, not injured", "")


def test_row_without_a_scheduled_game_is_dropped():
    """A bye week, or a schedule that doesn't cover the season being ingested."""
    rows = pd.DataFrame([_report(week=99)])
    assert build_snippets(rows, _schedule()) == []


def test_snippets_load_as_corpus_records():
    """The ingest output must satisfy the Snippet contract the retriever loads."""
    snippets = build_snippets(pd.DataFrame([_report()]), _schedule())
    assert [rag.Snippet(**s) for s in snippets]


def test_corpora_are_additive_and_deduplicated_by_id(tmp_path: Path):
    def corpus(name: str, rows: list[dict]) -> None:
        (tmp_path / name).write_text(json.dumps({"snippets": rows}))

    base = {"player": "P", "team": "T", "published": "2025-09-21", "source": "X"}
    corpus("a.json", [{**base, "id": "dup", "text": "from a"},
                      {**base, "id": "only-a", "text": "a"}])
    corpus("b.json", [{**base, "id": "dup", "text": "from b"}])

    loaded = {s.id: s.text for s in rag.load_snippets(tmp_path)}
    assert loaded == {"dup": "from b", "only-a": "a"}  # later file wins


def test_unreadable_corpus_does_not_take_retrieval_down(tmp_path: Path):
    (tmp_path / "bad.json").write_text("{not json")
    (tmp_path / "good.json").write_text(json.dumps({"snippets": [
        {"id": "g1", "player": "P", "team": "T", "published": "2025-09-21",
         "source": "X", "text": "fine"},
    ]}))
    assert [s.id for s in rag.load_snippets(tmp_path)] == ["g1"]


def test_recency_breaks_similarity_ties():
    """Same player, two weeks, near-identical text -- the newer one must win."""
    def snippet(sid: str, published: str) -> rag.Snippet:
        return rag.Snippet(id=sid, player="Test Player", team="BAL",
                           published=published, source=SOURCE,
                           text="Test Player is listed as Questionable with a knee injury.")

    store = rag.InMemoryStore([snippet("old", "2024-11-03"), snippet("new", "2025-11-02")])
    [top, _] = store.search("Test Player BAL injury status outlook", k=2)
    assert top.id == "new"


# --- preseason ingest ------------------------------------------------------
#
# Before week 1 there are no box scores and no injury report. These cover the
# feeds that do exist, and the status codes the report spells out rather than
# repeats raw.

from datetime import date  # noqa: E402

import ingest_preseason as pre  # noqa: E402


def _chart_row(**over):
    row = {"player_name": "Trey McBride", "team": "ARI", "pos_abb": "TE",
           "pos_rank": 1, "week": 1, "season": 2026}
    row.update(over)
    return pd.Series(row)


def test_depth_chart_row_reads_as_a_sentence():
    text = pre.describe(_chart_row(), "ACT")
    assert "first-string TE on ARI's depth chart" in text
    assert "week 1 of the 2026 season" in text
    assert "on the active roster" in text


def test_reserve_status_is_spelled_out_not_left_as_a_code():
    text = pre.describe(_chart_row(), "RES")
    assert "RES" not in text, "raw status code leaked into prose"
    assert "reserve" in text and "unavailable" in text


def test_unknown_status_code_is_reported_verbatim_not_guessed():
    text = pre.describe(_chart_row(), "ZZZ")
    assert "Roster status: ZZZ." in text


def test_preseason_snippet_says_no_games_have_been_played():
    # The narrator must not present camp positioning as in-season form.
    assert "No games have been played yet" in pre.describe(_chart_row(), "ACT")


def test_missing_rank_does_not_produce_a_broken_sentence():
    text = pre.describe(_chart_row(pos_rank=float("nan")), None)
    assert "None" not in text and "nan" not in text


# --- refresh scheduling ----------------------------------------------------

import refresh  # noqa: E402


@pytest.mark.parametrize(
    "today, expected",
    [
        (date(2026, 9, 1), 2026),   # camp: the 2026 season is the one coming
        (date(2026, 12, 20), 2026),  # in season
        (date(2027, 1, 15), 2026),   # January still belongs to 2026
        (date(2027, 2, 8), 2026),    # so does the Super Bowl
        (date(2027, 3, 1), 2027),    # new league year
    ],
)
def test_current_season_rolls_over_in_march_not_january(today, expected):
    assert refresh.current_season(today) == expected


def test_refresh_only_drops_the_season_it_was_asked_about(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("weekly_2024.parquet", "weekly_2026.parquet", "injuries_2026.parquet"):
        (raw / name).write_bytes(b"x")
    monkeypatch.setattr(refresh, "RAW", raw)

    refresh.drop_cache(2026)

    assert (raw / "weekly_2024.parquet").exists(), "a completed season was re-downloaded"
    assert not (raw / "weekly_2026.parquet").exists()
    assert not (raw / "injuries_2026.parquet").exists()
