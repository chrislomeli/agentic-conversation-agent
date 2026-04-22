"""pg_gateway.py — Postgres + pgvector access layer.

Provides PgGateway, a psycopg3 connection-pool wrapper with two tiers of helpers:

    Tier 1 — raw SQL primitives:
        fetch_rows()        — SELECT → list[dict]
        execute()           — INSERT / UPDATE / DELETE → rowcount

    Tier 2 — entity upserts (used by the write-through wrappers):
        ensure_session()    — sessions (idempotent)
        upsert_exchanges()  — exchanges (bulk)
        upsert_thread()     — threads + thread_exchanges junction
        upsert_fragment()   — fragments + fragment_exchanges junction (embedding optional)
        upsert_profile()    — user_profiles (single row)

    Plus the pgvector search path:
        search_similar()    — cosine top-k on fragments.embedding

Dependencies (add to pyproject.toml):
    psycopg[binary]>=3.2
    psycopg-pool>=3.2
    pgvector>=0.3
    numpy>=1.26

Schema lives in data/schema.sql — run once:
    psql $POSTGRES_URL -f data/schema.sql
"""

import json
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

import numpy as np
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from journal_agent.configure.settings import get_settings
from journal_agent.model.session import Exchange, Fragment, Role, Tag, ThreadSegment, Turn, UserProfile, Insight

logger = logging.getLogger(__name__)

# Must match data/schema.sql vector(N). 384 = sentence-transformers/all-MiniLM-L6-v2
# (the fastembed default used by Embedder). Change both if you swap models.
EMBEDDING_DIM = 384


class PgGateway:
    """Postgres + pgvector access layer backed by a connection pool.

    Create once at app startup and share across graph nodes.
    """

    def __init__(self, min_size: int = 2, max_size: int = 10):
        url = get_settings().postgres_url
        # open=False defers actual connections until first use so import-time
        # startup doesn't fail if Postgres isn't running.
        self._pool = ConnectionPool(
            conninfo=url,
            min_size=min_size,
            max_size=max_size,
            kwargs={"row_factory": dict_row},
            open=False,
        )

    def open(self) -> None:
        """Open the pool. Call once at app startup."""
        self._pool.open(wait=True)
        logger.info("PgGateway pool open (min=%d max=%d)", self._pool.min_size, self._pool.max_size)

    def close(self) -> None:
        """Drain the pool. Call at app shutdown."""
        self._pool.close()

    @contextmanager
    def _conn(self) -> Generator:
        """Borrow a connection from the pool; psycopg3 auto-commits on clean exit."""
        with self._pool.connection() as conn:
            yield conn

    # ══════════════════════════════════════════════════════════════════════════
    # Tier 1 — raw SQL primitives
    # ══════════════════════════════════════════════════════════════════════════

    def fetch_rows(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        """Run a SELECT; return every row as a dict."""
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def execute(self, sql: str, params: tuple = ()) -> int:
        """Run a mutating statement; return rowcount."""
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.rowcount

    # ══════════════════════════════════════════════════════════════════════════
    # Tier 2 — entity upserts (write-through targets)
    # ══════════════════════════════════════════════════════════════════════════

    def ensure_session(self, session_id: str) -> None:
        """Create a sessions row if it doesn't exist. Safe to call repeatedly."""
        self.execute(
            "INSERT INTO sessions (session_id) VALUES (%s) ON CONFLICT DO NOTHING",
            (session_id,),
        )

    def upsert_exchanges(self, session_id: str, exchanges: list[Exchange]) -> int:
        """Bulk-upsert a session's Exchange records.

        Ensures the parent sessions row exists first. Returns the rowcount
        from the exchange insert.
        """
        if not exchanges:
            return 0
        self.ensure_session(session_id)

        sql = """
            INSERT INTO exchanges (exchange_id, session_id, timestamp, human_content, ai_content)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (exchange_id) DO UPDATE SET
                timestamp     = EXCLUDED.timestamp,
                human_content = EXCLUDED.human_content,
                ai_content    = EXCLUDED.ai_content
        """
        rows = [
            (
                ex.exchange_id,
                session_id,
                ex.timestamp,
                ex.human.content if ex.human else None,
                ex.ai.content if ex.ai else None,
            )
            for ex in exchanges
        ]
        with self._conn() as conn, conn.cursor() as cur:
            cur.executemany(sql, rows)
            return cur.rowcount

    def upsert_thread(self, session_id: str, thread: ThreadSegment) -> None:
        """Upsert a ThreadSegment + its junction rows.

        Called twice in the normal pipeline: once by save_threads (tags empty)
        and once by save_classified_threads (tags populated). COALESCE on tags
        means a second write with tags overlays the first, but re-running
        save_threads later won't null them out.
        """
        self.ensure_session(session_id)
        thread_id = f"{session_id}:{thread.thread_name}"
        tags_payload = (
            Jsonb([t.model_dump() for t in thread.tags]) if thread.tags else None
        )

        self.execute(
            """
            INSERT INTO threads (thread_id, session_id, thread_name, tags)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (thread_id) DO UPDATE SET
                tags = COALESCE(EXCLUDED.tags, threads.tags)
            """,
            (thread_id, session_id, thread.thread_name, tags_payload),
        )

        # Junction rows. Wipe & replace keeps positions in sync if the thread
        # gets re-decomposed; acceptable because upserts are idempotent per run.
        self.execute(
            "DELETE FROM thread_exchanges WHERE thread_id = %s", (thread_id,)
        )
        if thread.exchange_ids:
            with self._conn() as conn, conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO thread_exchanges (thread_id, exchange_id, position) "
                    "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                    [(thread_id, eid, i) for i, eid in enumerate(thread.exchange_ids)],
                )

    def upsert_fragment(
        self,
        fragment: Fragment,
        embedding: np.ndarray | None = None,
    ) -> None:
        """Upsert a Fragment + its junction rows.

        `embedding` is optional — callers that don't need vector search (e.g.
        backfills) can pass None and the column is left NULL.
        """
        self.ensure_session(fragment.session_id)
        tags_payload = (
            Jsonb([t.model_dump() for t in fragment.tags]) if fragment.tags else None
        )

        self.execute(
            """
            INSERT INTO fragments (fragment_id, session_id, content, tags, embedding, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (fragment_id) DO UPDATE SET
                content   = EXCLUDED.content,
                tags      = EXCLUDED.tags,
                embedding = COALESCE(EXCLUDED.embedding, fragments.embedding),
                timestamp = EXCLUDED.timestamp
            """,
            (
                fragment.fragment_id,
                fragment.session_id,
                fragment.content,
                tags_payload,
                embedding,
                fragment.timestamp,
            ),
        )

        self.execute(
            "DELETE FROM fragment_exchanges WHERE fragment_id = %s",
            (fragment.fragment_id,),
        )
        if fragment.exchange_ids:
            with self._conn() as conn, conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO fragment_exchanges (fragment_id, exchange_id) "
                    "VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    [(fragment.fragment_id, eid) for eid in fragment.exchange_ids],
                )

    def upsert_profile(self, profile: UserProfile) -> None:
        """Upsert the single user profile row."""
        self.execute(
            """
            INSERT INTO user_profiles (
                user_id, response_style, explanation_depth, tone,
                decision_style, learning_style, interests, pet_peeves,
                active_projects, recurring_themes, human_label, ai_label, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id) DO UPDATE SET
                response_style    = EXCLUDED.response_style,
                explanation_depth = EXCLUDED.explanation_depth,
                tone              = EXCLUDED.tone,
                decision_style    = EXCLUDED.decision_style,
                learning_style    = EXCLUDED.learning_style,
                interests         = EXCLUDED.interests,
                pet_peeves        = EXCLUDED.pet_peeves,
                active_projects   = EXCLUDED.active_projects,
                recurring_themes  = EXCLUDED.recurring_themes,
                human_label       = EXCLUDED.human_label,
                ai_label          = EXCLUDED.ai_label,
                updated_at        = EXCLUDED.updated_at
            """,
            (
                profile.user_id,
                profile.response_style,
                profile.explanation_depth,
                profile.tone,
                profile.decision_style,
                profile.learning_style,
                Jsonb(profile.interests),
                Jsonb(profile.pet_peeves),
                Jsonb(profile.active_projects),
                Jsonb(profile.recurring_themes),
                profile.human_label,
                profile.ai_label,
                profile.updated_at,
            ),
        )

    def upsert_insights(self, insights: list[Insight]) -> None:
        if not insights:
            return

        expected_insights = len(insights)

        # Upsert insights
        sql = """
              INSERT INTO insights (insight_id, label, body, verifier_status, confidence)
              VALUES (%s, %s, %s, %s, %s)
              ON CONFLICT (insight_id) DO UPDATE SET label           = EXCLUDED.label,
                                                     body            = EXCLUDED.body,
                                                     verifier_status = EXCLUDED.verifier_status,
                                                     confidence      = EXCLUDED.confidence; \
              """
        rows = [
            (i.insight_id, i.label, i.body, i.verifier_status, i.confidence)
            for i in insights
        ]
        with self._conn() as conn, conn.cursor() as cur:
            cur.executemany(sql, rows)
            if cur.rowcount != expected_insights:
                raise RuntimeError(
                    f"upsert_insights: expected {expected_insights} rows, got {cur.rowcount}"
                )

        # Upsert junction rows
        junction_rows = [
            (i.insight_id, f)
            for i in insights
            for f in i.fragment_ids
        ]

        if junction_rows:
            expected_junction = len(junction_rows)
            sql = """
                  INSERT INTO insight_fragments (insight_id, fragment_id)
                  VALUES (%s, %s)
                  ON CONFLICT DO NOTHING \
                  """
            with self._conn() as conn, conn.cursor() as cur:
                cur.executemany(sql, junction_rows)
                if cur.rowcount != expected_junction:
                    raise RuntimeError(
                        f"upsert_insights junction: expected {expected_junction} rows, got {cur.rowcount}"
                    )


    def fetch_profile(self, user_id: str = "default") -> UserProfile:
        """Return the user_profiles row for *user_id* as a list (0 or 1 items).

        Returns [] on miss or error so callers can do `rows[0] if rows else None`.
        """
        try:
            rows = self.fetch_rows(
                """
                SELECT user_id, response_style, explanation_depth, tone,
                       decision_style, learning_style, interests, pet_peeves,
                       active_projects, recurring_themes, human_label, ai_label, updated_at
                FROM user_profiles
                WHERE user_id = %s
                """,
                (user_id,),
            )
            if not rows:
                return []
            r = rows[0]
            return UserProfile(
                response_style=r["response_style"],
                explanation_depth=r["explanation_depth"],
                tone=r["tone"],
                decision_style=r["decision_style"],
                learning_style=r["learning_style"],
                interests=r["interests"] or [],
                pet_peeves=r["pet_peeves"] or [],
                active_projects=r["active_projects"] or [],
                recurring_themes=r["recurring_themes"] or [],
                human_label=r["human_label"],
                ai_label=r["ai_label"],
                updated_at=r["updated_at"],
            )
        except Exception:
            logger.exception("fetch_profile failed for user_id=%s", user_id)
            return []

    def fetch_exchanges(self, session_id: str) -> list[Exchange]:
        """Return all Exchange records for *session_id*, ordered by timestamp.

        Returns [] on miss or error.
        """
        try:
            rows = self.fetch_rows(
                """
                SELECT exchange_id, session_id, timestamp, human_content, ai_content
                FROM exchanges
                WHERE session_id = %s
                ORDER BY timestamp
                """,
                (session_id,),
            )
            if not rows:
                return []
            results = []
            for r in rows:
                results.append(Exchange(
                    exchange_id=r["exchange_id"],
                    session_id=r["session_id"],
                    timestamp=r["timestamp"],
                    human=Turn(session_id=r["session_id"], role=Role.HUMAN, content=r["human_content"])
                          if r["human_content"] else None,
                    ai=Turn(session_id=r["session_id"], role=Role.AI, content=r["ai_content"])
                       if r["ai_content"] else None,
                ))
            return results
        except Exception:
            logger.exception("fetch_exchanges failed for session_id=%s", session_id)
            return []

    def fetch_threads(self, session_id: str) -> list[ThreadSegment]:
        """Return ThreadSegments for *session_id* with exchange_ids from the junction.

        Returns [] on miss or error.
        """
        try:
            rows = self.fetch_rows(
                """
                SELECT
                    t.thread_name,
                    t.tags,
                    COALESCE(
                        array_agg(te.exchange_id ORDER BY te.position)
                        FILTER (WHERE te.exchange_id IS NOT NULL),
                        ARRAY[]::text[]
                    ) AS exchange_ids
                FROM threads t
                LEFT JOIN thread_exchanges te ON te.thread_id = t.thread_id
                WHERE t.session_id = %s
                GROUP BY t.thread_id, t.thread_name, t.tags
                ORDER BY t.thread_name
                """,
                (session_id,),
            )
            if not rows:
                return []
            results = []
            for r in rows:
                tag_data = r["tags"] if isinstance(r["tags"], list) else []
                results.append(ThreadSegment(
                    thread_name=r["thread_name"],
                    exchange_ids=list(r["exchange_ids"]) if r["exchange_ids"] else [],
                    tags=[Tag(**t) if isinstance(t, dict) else t for t in tag_data],
                ))
            return results
        except Exception:
            logger.exception("fetch_threads failed for session_id=%s", session_id)
            return []

    def get_last_session_id(self) -> str | None:
        """Return the session_id of the most recently started session, or None."""
        try:
            rows = self.fetch_rows(
                "SELECT session_id FROM sessions ORDER BY started_at DESC LIMIT 1"
            )
            return rows[0]["session_id"] if rows else None
        except Exception:
            logger.exception("get_last_session_id failed")
            return None
    
    
    def fetch_fragments(self, session_id: str | None = None) -> list[Fragment]:
        """Load fragments from Postgres, optionally filtered by session.

        Returns [] on miss or error.
        """
        try:
            where = "WHERE f.session_id = %s" if session_id else ""
            params = (session_id,) if session_id else ()
            sql = f"""
                SELECT
                    f.fragment_id,
                    f.session_id,
                    f.content,
                    f.tags,
                    f.timestamp,
                    COALESCE(
                        array_agg(fe.exchange_id) FILTER (WHERE fe.exchange_id IS NOT NULL),
                        ARRAY[]::text[]
                    ) AS exchange_ids
                FROM fragments f
                LEFT JOIN fragment_exchanges fe ON fe.fragment_id = f.fragment_id
                {where}
                GROUP BY f.fragment_id, f.session_id, f.content, f.tags, f.timestamp
                ORDER BY f.timestamp
            """
            rows = self.fetch_rows(sql, params)
            if not rows:
                return []
            results = []
            for r in rows:
                tag_data = r.get("tags") if isinstance(r.get("tags"), list) else []
                results.append(Fragment(
                    fragment_id=r["fragment_id"],
                    session_id=r["session_id"],
                    content=r["content"],
                    exchange_ids=list(r["exchange_ids"]) if r["exchange_ids"] else [],
                    tags=[Tag(**t) if isinstance(t, dict) else t for t in tag_data],
                    timestamp=r["timestamp"],
                ))
            return results
        except Exception:
            logger.exception("fetch_fragments failed for session_id=%s", session_id)
            return []

    def fetch_insights(self, label: str | None = None, to_date: datetime | None = None) -> list[Insight]:
        """Return Insights with their associated fragment_ids from the junction table."""
        try:
            rows = self.fetch_rows("""
                                   SELECT t.insight_id,
                                          t.label,
                                          t.body,
                                          t.verifier_status,
                                          t.confidence,
                                          t.created_at,
                                          COALESCE(
                                                          array_agg(te.fragment_id)
                                                          FILTER (WHERE te.fragment_id IS NOT NULL),
                                                          ARRAY []::text[]
                                          ) AS fragment_ids
                                   FROM insights t
                                            LEFT JOIN insight_fragments te ON te.insight_id = t.insight_id
                                   WHERE (%s IS NULL OR t.label = %s)
                                     AND (%s IS NULL OR t.created_at <= %s)
                                   GROUP BY t.insight_id, t.label, t.body, t.verifier_status, t.confidence, t.created_at
                                   ORDER BY t.created_at;
                                   """,
                                   (label, label, to_date, to_date)
                                   )
            if not rows:
                return []
            return [
                Insight(
                    insight_id=r["insight_id"],
                    label=r["label"],
                    body=r["body"],
                    verifier_status=r["verifier_status"],
                    confidence=r["confidence"],
                    created_at=r["created_at"],
                    fragment_ids=list(r["fragment_ids"]) if r["fragment_ids"] else []
                )
                for r in rows
            ]
        except Exception:
            logger.exception("fetch_insights failed for label=%s, to_date=%s", label, to_date)
            return []


    # ══════════════════════════════════════════════════════════════════════════
    # pgvector search
    # ══════════════════════════════════════════════════════════════════════════

    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> list[tuple[Fragment, float]]:
        """Top-k cosine-similar fragments.

        Uses pgvector's `<=>` cosine-distance operator. Distance ∈ [0, 2];
        we convert to a 0–1 relevance score (`1 - distance / 2`).
        Exchange_ids are reconstructed from the fragment_exchanges junction.
        """
        sql = """
            SELECT
                f.fragment_id,
                f.session_id,
                f.content,
                f.tags,
                f.timestamp,
                COALESCE(
                    (SELECT array_agg(fe.exchange_id)
                     FROM fragment_exchanges fe
                     WHERE fe.fragment_id = f.fragment_id),
                    ARRAY[]::text[]
                ) AS exchange_ids,
                1.0 - (f.embedding <=> %s::vector) / 2.0 AS score
            FROM fragments f
            WHERE f.embedding IS NOT NULL
            ORDER BY f.embedding <=> %s::vector
            LIMIT %s
        """
        try:
            rows = self.fetch_rows(sql, (query_embedding, query_embedding, top_k))
            if not rows:
                return []
            results: list[tuple[Fragment, float]] = []
            for r in rows:
                score = r["score"]
                if score < min_score:
                    continue
                tag_data = r.get("tags") if isinstance(r.get("tags"), list) else []
                fragment = Fragment(
                    fragment_id=r["fragment_id"],
                    session_id=r["session_id"],
                    content=r["content"],
                    exchange_ids=list(r["exchange_ids"]) if r["exchange_ids"] else [],
                    tags=[Tag(**t) if isinstance(t, dict) else t for t in tag_data],
                    timestamp=r["timestamp"] or datetime.now(),
                )
                results.append((fragment, score))
            return results
        except Exception:
            logger.exception("search_similar failed")
            return []


# ── Module-level singleton (lazy) ──────────────────────────────────────────────

_gateway: PgGateway | None = None


def get_pg_gateway() -> PgGateway:
    """Return the module-level PgGateway singleton, opening the pool on first call."""
    global _gateway
    if _gateway is None:
        _gateway = PgGateway()
        _gateway.open()
    return _gateway
