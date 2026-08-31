-- Postgres schema: feature store + pgvector news store in one instance.
--
-- Both paths are optional for the demo -- with no DATABASE_URL the API reads
-- features from the cached parquet and retrieves from an in-process index. This
-- schema is what the production path looks like.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- Feature store: one row per projectable player-week.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS player_week_features (
    player_id              TEXT        NOT NULL,
    season                 SMALLINT    NOT NULL,
    week                   SMALLINT    NOT NULL,
    player_name            TEXT        NOT NULL,
    position               TEXT        NOT NULL,
    team                   TEXT        NOT NULL,
    opponent               TEXT        NOT NULL,

    -- Pre-kickoff context features (see model/features.py CONTEXT_FEATURES).
    is_home                REAL        NOT NULL,
    rest_days              REAL        NOT NULL,
    games_played           REAL        NOT NULL,
    roll3_ppr              REAL,
    roll5_ppr              REAL,
    roll3_targets          REAL,
    roll3_carries          REAL,
    roll5_target_share     REAL,
    opp_def_allowed_prior  REAL,

    -- The last-N-game window the sequence model consumes, stored as JSON so the
    -- shape stays in lockstep with SEQ_STATS rather than being frozen into DDL.
    sequence               JSONB       NOT NULL,

    -- Outcome. NULL until the game is played -- never read at projection time.
    actual_ppr             REAL,

    updated_at             TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (player_id, season, week)
);

-- The serving query is always "this player, this week" or "everyone this week".
CREATE INDEX IF NOT EXISTS idx_features_week ON player_week_features (season, week);
CREATE INDEX IF NOT EXISTS idx_features_name ON player_week_features (lower(player_name));

-- ---------------------------------------------------------------------------
-- Vector store: recent injury/news snippets for retrieval.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS news_snippets (
    id         TEXT PRIMARY KEY,
    player     TEXT NOT NULL,
    team       TEXT,
    published  DATE,
    source     TEXT,
    text       TEXT NOT NULL,
    embedding  vector(512)   -- must match api/rag.py EMBED_DIM
);

-- Cosine distance, matching the `<=>` operator used by the retriever.
CREATE INDEX IF NOT EXISTS idx_news_embedding
    ON news_snippets USING ivfflat (embedding vector_cosine_ops) WITH (lists = 20);

CREATE INDEX IF NOT EXISTS idx_news_player ON news_snippets (lower(player));
