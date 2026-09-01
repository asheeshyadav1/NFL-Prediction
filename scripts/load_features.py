"""Load the feature store and news corpus into Postgres.

    DATABASE_URL=postgresql://... python scripts/load_features.py

Optional -- the API falls back to cached parquet and an in-memory index. Applies
`infra/sql/schema.sql`, writes one row per player-week, embeds the news corpus.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from api import rag  # noqa: E402
from api.feature_store import FIRST_SEASON, LAST_SEASON  # noqa: E402
from data import load_schedules, load_weekly  # noqa: E402
from dataset import build_windows  # noqa: E402
from features import CONTEXT_FEATURES, build  # noqa: E402

log = logging.getLogger("load-features")
SCHEMA = ROOT / "infra" / "sql" / "schema.sql"

# Context columns that have their own typed column in the schema.
STORED_CONTEXT = [
    "is_home", "rest_days", "games_played", "roll3_ppr", "roll5_ppr",
    "roll3_targets", "roll3_carries", "roll5_target_share", "opp_def_allowed_prior",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=2000)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("DATABASE_URL is not set -- nothing to load into")

    import psycopg
    from pgvector.psycopg import register_vector

    years = list(range(FIRST_SEASON, LAST_SEASON + 1))
    log.info("building features for %s-%s ...", years[0], years[-1])
    frame = build(load_weekly(years), load_schedules(years))
    windows = build_windows(frame)
    kept = windows["frame"].reset_index(drop=True)
    log.info("prepared %s player-weeks", f"{len(kept):,}")

    with psycopg.connect(dsn, autocommit=False) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            log.info("applying schema ...")
            cur.execute(SCHEMA.read_text())

            rows = []
            for i in range(len(kept)):
                r = kept.iloc[i]
                rows.append(
                    (
                        r["player_id"], int(r["season"]), int(r["week"]),
                        r["player_display_name"], r["position"],
                        r["recent_team"], r["opponent_team"],
                        *[float(r[c]) for c in STORED_CONTEXT],
                        json.dumps(windows["seq"][i].tolist()),
                        float(windows["y"][i]),
                    )
                )

            placeholders = ", ".join(["%s"] * (7 + len(STORED_CONTEXT) + 2))
            statement = f"""
                INSERT INTO player_week_features (
                    player_id, season, week, player_name, position, team, opponent,
                    {", ".join(STORED_CONTEXT)}, sequence, actual_ppr
                ) VALUES ({placeholders})
                ON CONFLICT (player_id, season, week) DO UPDATE SET
                    sequence = EXCLUDED.sequence,
                    actual_ppr = EXCLUDED.actual_ppr,
                    updated_at = now()
            """
            for start in range(0, len(rows), args.batch_size):
                cur.executemany(statement, rows[start : start + args.batch_size])
                log.info("  loaded %s / %s", min(start + args.batch_size, len(rows)), len(rows))

            snippets = rag.load_snippets()
            vectors = rag.embed([s.document() for s in snippets])
            for snippet, vec in zip(snippets, vectors):
                cur.execute(
                    """
                    INSERT INTO news_snippets (id, player, team, published, source, text, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        text = EXCLUDED.text, embedding = EXCLUDED.embedding
                    """,
                    (snippet.id, snippet.player, snippet.team, snippet.published,
                     snippet.source, snippet.text, vec),
                )
            log.info("loaded %d news snippets", len(snippets))
        conn.commit()

    # Sanity check that unused context features aren't silently dropped.
    missing = set(CONTEXT_FEATURES) - set(STORED_CONTEXT) - {"week", "is_QB", "is_RB", "is_WR", "is_TE"}
    if missing:
        log.warning("context features not persisted: %s", sorted(missing))
    log.info("done")


if __name__ == "__main__":
    main()
