"""Ingest the official NFL injury report into the retrieval corpus.

    python scripts/ingest_injuries.py --seasons 2025 --last-weeks 4

Writes `data/news/injuries.json` in the demo fixture's schema, so the retriever
picks it up unchanged. Every snippet carries `source = "NFL-INJURY-REPORT"`, so
real reporting stays distinguishable from the synthetic fixture at citation
time. Source is the nflverse `injuries` release, which carries `gsis_id`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

from data import load_injuries, load_schedules  # noqa: E402

log = logging.getLogger("ingest-injuries")

OUT_PATH = ROOT / "data" / "news" / "injuries.json"
SOURCE = "NFL-INJURY-REPORT"

# The projectable positions. Everyone else appears on the report but can never
# be the subject of a start/sit query, and would only dilute retrieval.
POSITIONS = ("QB", "RB", "WR", "TE")


def _blank(value) -> bool:
    return value is None or pd.isna(value) or not str(value).strip()


# The report uses this free-text value where an injury would go, to mean a
# healthy scratch. Rendering it as "a not injury related - resting player injury"
# is both wrong and the kind of thing an LLM will happily repeat.
_NOT_AN_INJURY = "not injury related"


def _injury_phrase(primary, secondary) -> str:
    """The injuries named on a report row, or "" if it names none."""
    parts = [
        str(p).strip().lower()
        for p in (primary, secondary)
        if not _blank(p) and _NOT_AN_INJURY not in str(p).strip().lower()
    ]
    return " and ".join(parts)


def _rested(row) -> bool:
    return any(
        _NOT_AN_INJURY in str(row.get(col)).strip().lower()
        for col in ("report_primary_injury", "practice_primary_injury")
        if not _blank(row.get(col))
    )


def describe(row: pd.Series) -> str:
    """One report row as a sentence.

    Both halves of the report are kept. Game status is what fantasy managers act
    on, but it is blank for most of the week, and practice participation is the
    only signal that exists on a Wednesday -- dropping it would leave the
    retriever with nothing to return for exactly the players being asked about.
    """
    status = None if _blank(row.get("report_status")) else str(row["report_status"]).strip()
    injury = _injury_phrase(row.get("report_primary_injury"), row.get("report_secondary_injury"))
    practice = (
        None if _blank(row.get("practice_status")) else str(row["practice_status"]).strip()
    )
    practice_injury = _injury_phrase(row.get("practice_primary_injury"), None)

    sentences = []
    if status:
        sentences.append(
            f"{row['full_name']} is listed as {status}"
            + (f" with a {injury} injury." if injury else " on the injury report.")
        )
    elif injury or practice_injury:
        sentences.append(
            f"{row['full_name']} carries no game-status designation but is on the "
            f"report with a {injury or practice_injury} injury."
        )
    elif _rested(row):
        sentences.append(
            f"{row['full_name']} appears on the report as rested, not injured "
            f"(no injury designation)."
        )
    else:
        sentences.append(f"{row['full_name']} appears on the injury report.")

    if practice:
        sentences.append(f"Practice participation: {practice.lower()}.")

    sentences.append(
        f"Week {int(row['week'])} of the {int(row['season'])} season, {row['team']}."
    )
    return " ".join(sentences)


def build_snippets(injuries: pd.DataFrame, schedules: pd.DataFrame) -> list[dict]:
    """Report rows -> corpus records, newest first.

    `published` comes from the schedule rather than the report, which carries no
    date of its own. Dating a snippet by its game means a citation shown to the
    user reads as the week it actually applies to.
    """
    if injuries.empty:
        return []

    df = injuries[injuries["position"].isin(POSITIONS)].copy()
    # Rows with neither a game status nor a practice status carry no signal.
    df = df[~(df["report_status"].isna() & df["practice_status"].isna())]
    df = df.merge(_gamedays(schedules), on=["season", "week", "team"], how="left")

    records = [
        {
            "id": f"inj-{int(r['season'])}-{int(r['week']):02d}-{r['gsis_id']}",
            "player": r["full_name"],
            "team": r["team"],
            "published": r["published"],
            # Carried explicitly so retrieval can scope to the week being asked
            # about. `published` alone cannot: it is a date, and the corpus now
            # spans ten seasons of them.
            "season": int(r["season"]),
            "week": int(r["week"]),
            "source": SOURCE,
            "text": describe(r),
        }
        for _, r in df.iterrows()
        if not _blank(r.get("gsis_id")) and not _blank(r.get("published"))
    ]
    records.sort(key=lambda s: (s["published"], s["id"]), reverse=True)
    return records


def _gamedays(schedules: pd.DataFrame) -> pd.DataFrame:
    """(season, week, team) -> game date, one row per team per week."""
    home = schedules.rename(columns={"home_team": "team"})[["season", "week", "team", "gameday"]]
    away = schedules.rename(columns={"away_team": "team"})[["season", "week", "team", "gameday"]]
    both = pd.concat([home, away], ignore_index=True)
    both["published"] = both["gameday"].astype(str)
    return both.drop_duplicates(["season", "week", "team"])[["season", "week", "team", "published"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=int, nargs="+", default=[2025],
                    help="seasons to ingest (default: 2025)")
    ap.add_argument("--last-weeks", type=int, default=0,
                    help="keep only the N most recent weeks of the latest season "
                         "(0 = keep everything)")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    injuries = load_injuries(args.seasons)
    if injuries.empty:
        log.error("no injury data for %s -- nothing written", args.seasons)
        return 1

    if args.last_weeks:
        latest = int(injuries["season"].max())
        cutoff = int(injuries[injuries["season"] == latest]["week"].max()) - args.last_weeks + 1
        injuries = injuries[(injuries["season"] == latest) & (injuries["week"] >= cutoff)]

    snippets = build_snippets(injuries, load_schedules(args.seasons))
    if not snippets:
        log.error("no snippets survived filtering -- nothing written")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "_README": (
            "REAL DATA. Generated by scripts/ingest_injuries.py from the official NFL "
            "injury report (nflverse `injuries` release -- the same practice and "
            "game-status report published at nfl.com/injuries). Regenerate rather "
            "than hand-editing."
        ),
        "_schema": "id, player, team, published, season, week, source, text",
        "_source_label": f"Every entry carries source = '{SOURCE}'.",
        "snippets": snippets,
    }, indent=2) + "\n")

    dates = [s["published"] for s in snippets]
    log.info("wrote %d snippets to %s (%d players, %s..%s)",
             len(snippets), args.out, len({s["player"] for s in snippets}),
             min(dates), max(dates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
