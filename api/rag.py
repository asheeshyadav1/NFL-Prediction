"""Retrieval over recent injury/news snippets.

Feeds context to the narrator and never touches the projection -- this module
has no access to the model and returns only text. Embeddings use a hashing
vectorizer (deterministic, no fitting step); swap `embed()` to change that.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

log = logging.getLogger(__name__)

EMBED_DIM = 512
NEWS_DIR = Path(__file__).resolve().parent.parent / "data" / "news"

_vectorizer = HashingVectorizer(
    n_features=EMBED_DIM, alternate_sign=False, norm="l2", stop_words="english"
)


def embed(texts: list[str]) -> np.ndarray:
    return np.asarray(_vectorizer.transform(texts).todense(), dtype=np.float32)


@dataclass(frozen=True)
class Snippet:
    id: str
    player: str
    team: str
    published: str
    text: str
    source: str
    score: float = 0.0

    def cite(self) -> str:
        return f"[{self.source} | {self.published}] {self.player}: {self.text}"

    def document(self) -> str:
        """What gets embedded. Name and team are the highest-signal terms for a
        start/sit query, so they go in rather than being left to their fields."""
        return f"{self.player} {self.team} {self.text}"


class InMemoryStore:
    """Cosine-similarity search in-process. Fallback when no database is
    configured; same interface as the pgvector store."""

    backend = "in-memory"

    def __init__(self, snippets: list[Snippet]) -> None:
        self._snippets = snippets
        self._matrix = (
            embed([s.document() for s in snippets]) if snippets else np.zeros((0, EMBED_DIM))
        )

    def search(self, query: str, k: int = 4) -> list[Snippet]:
        if not self._snippets:
            return []
        scores = self._matrix @ embed([query])[0]  # rows are L2-normalised
        # Recency breaks ties: one player's Week 3 and Week 14 reports embed
        # almost identically, so similarity alone can return a stale status.
        # lexsort orders by its last key first -- score desc, then published desc.
        published = np.array([s.published for s in self._snippets])
        top = np.lexsort((published, scores))[::-1][:k]
        return [
            Snippet(**{**self._snippets[i].__dict__, "score": float(scores[i])})
            for i in top
            if scores[i] > 0
        ]


class PgVectorStore:
    """pgvector-backed search. Used when DATABASE_URL is set."""

    backend = "pgvector"

    def __init__(self, dsn: str, snippets: list[Snippet]) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self._conn = psycopg.connect(dsn, autocommit=True)
        register_vector(self._conn)
        self._ensure_schema()
        self._upsert(snippets)

    def _ensure_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS news_snippets (
                    id         TEXT PRIMARY KEY,
                    player     TEXT NOT NULL,
                    team       TEXT,
                    published  DATE,
                    source     TEXT,
                    text       TEXT NOT NULL,
                    embedding  vector({EMBED_DIM})
                )
                """
            )

    def _upsert(self, snippets: list[Snippet]) -> None:
        if not snippets:
            return
        vectors = embed([s.document() for s in snippets])
        with self._conn.cursor() as cur:
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

    def search(self, query: str, k: int = 4) -> list[Snippet]:
        vec = embed([query])[0]
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, player, team, published, source, text,
                       1 - (embedding <=> %s) AS score
                FROM news_snippets
                ORDER BY embedding <=> %s, published DESC
                LIMIT %s
                """,
                (vec, vec, k),
            )
            return [
                Snippet(id=r[0], player=r[1], team=r[2], published=str(r[3]),
                        source=r[4], text=r[5], score=float(r[6]))
                for r in cur.fetchall()
            ]


def load_snippets(path: Path = NEWS_DIR) -> list[Snippet]:
    """Load every corpus under `path` (or a single file).

    Corpora are additive and each keeps its own `source` label, so a citation
    says where it came from. Ids dedupe across files, so re-ingesting overwrites.
    """
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    if not files:
        log.warning("no news corpus under %s -- retrieval will return nothing", path)
        return []

    snippets: dict[str, Snippet] = {}
    for file in files:
        try:
            rows = json.loads(file.read_text())["snippets"]
        except (ValueError, KeyError) as exc:  # a bad corpus shouldn't be fatal
            log.warning("skipping unreadable corpus %s (%s)", file.name, exc)
            continue
        for row in rows:
            snippet = Snippet(**row)
            snippets[snippet.id] = snippet
        log.info("loaded %d snippets from %s", len(rows), file.name)
    return list(snippets.values())


def build_store(snippets: list[Snippet] | None = None):
    """Return a pgvector store if DATABASE_URL is set, else the in-memory one."""
    snippets = load_snippets() if snippets is None else snippets
    dsn = os.environ.get("DATABASE_URL")
    if dsn:
        try:
            store = PgVectorStore(dsn, snippets)
            log.info("retrieval backend: pgvector (%d snippets)", len(snippets))
            return store
        except Exception as exc:  # a missing DB shouldn't take the API down
            log.warning("pgvector unavailable (%s) -- falling back to in-memory", exc)
    log.info("retrieval backend: in-memory (%d snippets)", len(snippets))
    return InMemoryStore(snippets)
