"""subjects_repo.py — Phase 11 repository for subjects, claims, votes, and processing.

One repository owns all four Phase 11 tables because they form one consistent
unit: subjects own claims, votes attach to subjects, fragment_processing is
the bookkeeping record for a vote-producing run. Splitting them across repos
would create transactional ambiguity ("did the votes save but the claim
not?"), which is exactly the kind of thing the reflect node should not have
to reason about.

Method bodies are NotImplementedError in the skeleton — the contract (names,
signatures, return types, intended SQL) is what's being committed here. Real
implementations land in PR2.

Design doc: design/phase11-claim-based-insights.md
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from psycopg.types.json import Jsonb

from journal_agent.model.insights import (
    Claim,
    FragmentProcessing,
    ProcessingStatus,
    ProposedSubject,
    Stance,
    Subject,
    SubjectStatus,
    Vote,
)
from journal_agent.stores.embedder import Embedder
from journal_agent.stores.pg_gateway import PgGateway

logger = logging.getLogger(__name__)


class SubjectsRepository:
    """All Phase 11 persistence: subjects, claims, votes, and fragment_processing.

    Reads, writes, and the small handful of derived queries (most-recent-claim,
    candidate routing by similarity, processed-fragment lookups) live here.
    Aggregation/scoring over votes is policy and lives in
    `graph.nodes.insight_nodes.compute_traction`, not here — this repo just
    returns rows.
    """

    def __init__(
        self,
        pg_gateway: PgGateway,
        embedder: Embedder | None = None,
    ):
        self._pg = pg_gateway
        self._embedder = embedder or Embedder()

    # ── private row-mapping helpers ───────────────────────────────────────

    def _row_to_subject(self, r: dict, created_key: str = "created_at") -> Subject:
        return Subject(
            subject_id=r["subject_id"],
            label=r["label"],
            description=r.get("description"),
            status=SubjectStatus(r["status"]),
            parent_subject_id=r.get("parent_subject_id"),
            merged_into_id=r.get("merged_into_id"),
            created_at=r[created_key],
            last_activity_at=r["last_activity_at"],
        )

    def _row_to_claim(self, r: dict, text_key: str = "text", created_key: str = "created_at") -> Claim:
        raw_emb = r.get("embedding")
        if raw_emb is None:
            embedding: list[float] = []
        elif isinstance(raw_emb, np.ndarray):
            embedding = raw_emb.tolist()
        elif isinstance(raw_emb, str):
            embedding = json.loads(raw_emb)
        else:
            embedding = list(raw_emb)
        return Claim(
            claim_id=r["claim_id"],
            subject_id=r["subject_id"],
            text=r[text_key],
            version=r["version"],
            is_current=r["is_current"],
            embedding=embedding,
            regenerated_at_vote_count=r["regenerated_at_vote_count"],
            created_at=r[created_key],
        )

    def _row_to_vote(self, r: dict) -> Vote:
        return Vote(
            vote_id=r["vote_id"],
            subject_id=r["subject_id"],
            claim_id=r["claim_id"],
            fragment_id=r["fragment_id"],
            stance=Stance(r["stance"]),
            strength=float(r["strength"]),
            reasoning=r["reasoning"],
            fragment_dated_at=r["fragment_dated_at"],
            processed_at=r["processed_at"],
            model_signature=r["model_signature"],
            signals=r.get("signals"),
            invalidated_at=r.get("invalidated_at"),
            invalidation_reason=r.get("invalidation_reason"),
        )

    # ── subjects ─────────────────────────────────────────────────────────

    def create_subject_with_claim(
        self,
        proposed: ProposedSubject,
        first_vote: Vote,
    ) -> tuple[Subject, Claim, Vote]:
        """Create a new Subject, its first Claim (version=1, is_current=True), and
        persist the bundled first Vote, in one transaction.

        Real SQL (TODO):
            INSERT INTO subjects (...)
            INSERT INTO claims (..., version=1, is_current=TRUE)
                with embedding over `label || text`
            INSERT INTO votes (..., claim_id=<new claim id>)

        Returns the persisted records (with server-assigned IDs and timestamps).
        """
        now = datetime.now()
        subject = Subject(
            label=proposed.label,
            description=proposed.description,
            status=SubjectStatus.ACTIVE,
            created_at=now,
            last_activity_at=first_vote.fragment_dated_at,
        )

        vec = self._embedder.embed(f"{proposed.label} {proposed.initial_claim}")

        claim = Claim(
            subject_id=subject.subject_id,
            text=proposed.initial_claim,
            version=1,
            is_current=True,
            embedding=vec.tolist(),
            regenerated_at_vote_count=0,
            created_at=now,
        )

        vote = first_vote.model_copy(update={
            "subject_id": subject.subject_id,
            "claim_id": claim.claim_id,
        })

        with self._pg.conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO subjects (subject_id, label, description, status,
                        parent_subject_id, merged_into_id, created_at, last_activity_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        subject.subject_id, subject.label, subject.description,
                        subject.status.value, subject.parent_subject_id,
                        subject.merged_into_id, subject.created_at, subject.last_activity_at,
                    ),
                )
                cur.execute(
                    """
                    INSERT INTO claims (claim_id, subject_id, text, version, is_current,
                        embedding, regenerated_at_vote_count, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s::vector, %s, %s)
                    """,
                    (
                        claim.claim_id, claim.subject_id, claim.text, claim.version,
                        claim.is_current, vec.tolist(), claim.regenerated_at_vote_count,
                        claim.created_at,
                    ),
                )
                cur.execute(
                    """
                    INSERT INTO votes (vote_id, subject_id, claim_id, fragment_id, stance,
                        strength, reasoning, fragment_dated_at, processed_at, model_signature, signals)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        vote.vote_id, vote.subject_id, vote.claim_id, vote.fragment_id,
                        vote.stance.value, vote.strength, vote.reasoning,
                        vote.fragment_dated_at, vote.processed_at, vote.model_signature,
                        Jsonb(vote.signals) if vote.signals else None,
                    ),
                )

        return subject, claim, vote

    def get_subject(self, subject_id: str) -> Subject | None:
        """Fetch a single Subject by id. Returns None on miss."""
        rows = self._pg.fetch_rows(
            "SELECT * FROM subjects WHERE subject_id = %s",
            (subject_id,),
        )
        return self._row_to_subject(rows[0]) if rows else None

    def list_active_subjects(self, limit: int = 100) -> list[Subject]:
        """List subjects with status='active', most-recently-active first.

        Used by the subject_proposer to brief the LLM on existing subjects.
        """
        rows = self._pg.fetch_rows(
            """
            SELECT * FROM subjects WHERE status = 'active'
            ORDER BY last_activity_at DESC LIMIT %s
            """,
            (limit,),
        )
        return [self._row_to_subject(r) for r in rows]

    def mark_subject_status(self, subject_id: str, status: str) -> None:
        """Set subjects.status (e.g., dormant, superseded, merged).

        Status enum values: 'active' | 'dormant' | 'superseded' | 'merged'.
        """
        self._pg.execute(
            "UPDATE subjects SET status = %s WHERE subject_id = %s",
            (status, subject_id),
        )

    def count_active_subjects(self) -> int:
        """Cheap count of active subjects. Used by the cold-start gate in the
        claim reflection graph: when this drops below COLD_START_SUBJECT_THRESHOLD,
        the graph runs cluster_seed_subjects instead of the per-fragment path.
        """
        rows = self._pg.fetch_rows(
            "SELECT COUNT(*) AS cnt FROM subjects WHERE status = 'active'",
            (),
        )
        return int(rows[0]["cnt"]) if rows else 0

    def list_active_subjects_with_claims(self, limit: int = 100) -> list[tuple[Subject, Claim]]:
        """Return (Subject, current Claim) pairs for all active subjects.

        Used by propose_subject to show the LLM what is already tracked,
        so it does not propose a duplicate.
        """
        rows = self._pg.fetch_rows(
            """
            SELECT
                s.subject_id, s.label, s.description, s.status,
                s.parent_subject_id, s.merged_into_id,
                s.created_at        AS subject_created_at,
                s.last_activity_at,
                c.claim_id, c.text  AS claim_text, c.version, c.is_current,
                c.embedding, c.regenerated_at_vote_count,
                c.created_at        AS claim_created_at
            FROM subjects s
            JOIN claims c ON c.subject_id = s.subject_id AND c.is_current = TRUE
            WHERE s.status = 'active'
            ORDER BY s.last_activity_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        result = []
        for r in rows:
            subject = self._row_to_subject(r, created_key="subject_created_at")
            claim = self._row_to_claim(r, text_key="claim_text", created_key="claim_created_at")
            result.append((subject, claim))
        return result

    # ── claims ───────────────────────────────────────────────────────────

    def get_current_claim(self, subject_id: str) -> Claim | None:
        """Fetch the row in `claims` where subject_id matches and is_current=TRUE."""
        rows = self._pg.fetch_rows(
            "SELECT * FROM claims WHERE subject_id = %s AND is_current = TRUE",
            (subject_id,),
        )
        return self._row_to_claim(rows[0]) if rows else None

    def regenerate_claim(
        self,
        subject_id: str,
        new_text: str,
        regenerated_at_vote_count: int,
    ) -> Claim:
        """Append a new Claim version and flip is_current.

        SELECT FOR UPDATE on the subject row serialises concurrent regenerations:
        a second caller blocks at the lock until the first commits, then reads the
        already-incremented MAX(version). Version is computed in SQL so there is no
        window between reading it and writing it.

        Returns the new Claim. Old versions are retained for audit.
        """
        subject = self.get_subject(subject_id)
        if subject is None:
            raise ValueError(f"Subject not found: {subject_id}")

        # Embed outside the transaction — CPU work, no lock needed.
        vec = self._embedder.embed(f"{subject.label} {new_text}")
        now = datetime.now()
        new_claim_id = str(uuid.uuid4())

        with self._pg.conn() as conn:
            with conn.cursor() as cur:
                # Lock the subject row for the duration of this transaction.
                # Any concurrent call to regenerate_claim for this subject blocks here.
                cur.execute(
                    "SELECT subject_id FROM subjects WHERE subject_id = %s FOR UPDATE",
                    (subject_id,),
                )
                cur.execute(
                    "UPDATE claims SET is_current = FALSE WHERE subject_id = %s AND is_current = TRUE",
                    (subject_id,),
                )
                # Version is derived from the current max inside the same transaction,
                # so no concurrent caller can observe the same value.
                cur.execute(
                    """
                    INSERT INTO claims (claim_id, subject_id, text, version, is_current,
                        embedding, regenerated_at_vote_count, created_at)
                    SELECT %s, %s, %s, COALESCE(MAX(version), 0) + 1, TRUE, %s::vector, %s, %s
                    FROM claims WHERE subject_id = %s
                    RETURNING version
                    """,
                    (
                        new_claim_id, subject_id, new_text,
                        vec.tolist(), regenerated_at_vote_count, now,
                        subject_id,
                    ),
                )
                row = cur.fetchone()
                next_version = row["version"]

        return Claim(
            claim_id=new_claim_id,
            subject_id=subject_id,
            text=new_text,
            version=next_version,
            is_current=True,
            embedding=vec.tolist(),
            regenerated_at_vote_count=regenerated_at_vote_count,
            created_at=now,
        )

    def search_candidate_subjects(
        self,
        query_embedding: list[float],
        top_k: int,
        min_similarity: float,
    ) -> list[tuple[Subject, Claim, float]]:
        """Vector-search current claim embeddings; return top-K with their Subjects.

        Returns triples of (Subject, current Claim, cosine similarity score)
        sorted by similarity descending. Filters out claims whose similarity
        is below min_similarity and subjects whose status != 'active'.

        This is the entry point for route_candidates.
        """
        vec = np.array(query_embedding, dtype=np.float32).tolist()

        rows = self._pg.fetch_rows(
            """
            SELECT
                s.subject_id, s.label, s.description, s.status,
                s.parent_subject_id, s.merged_into_id,
                s.created_at        AS subject_created_at,
                s.last_activity_at,
                c.claim_id, c.text  AS claim_text, c.version, c.is_current, c.embedding,
                c.regenerated_at_vote_count,
                c.created_at        AS claim_created_at,
                1.0 - (c.embedding <=> %s::vector) / 2.0 AS similarity
            FROM claims c
            JOIN subjects s ON s.subject_id = c.subject_id
            WHERE c.is_current = TRUE
              AND s.status = 'active'
              AND c.embedding IS NOT NULL
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
            """,
            (vec, vec, top_k),
        )

        results: list[tuple[Subject, Claim, float]] = []
        for r in rows:
            similarity = float(r["similarity"])
            if similarity < min_similarity:
                continue
            subject = self._row_to_subject(r, created_key="subject_created_at")
            claim = self._row_to_claim(r, text_key="claim_text", created_key="claim_created_at")
            results.append((subject, claim, similarity))

        return results

    # ── votes ────────────────────────────────────────────────────────────

    def insert_votes(self, votes: list[Vote]) -> None:
        """Bulk-insert votes; updates each subject's last_activity_at to max(
        current, vote.fragment_dated_at).

        No-op on empty list. Idempotent against the
        (subject_id, fragment_id, stance) unique-active constraint —
        duplicates are silently dropped via ON CONFLICT DO NOTHING.
        """
        if not votes:
            return

        rows = [
            (
                v.vote_id, v.subject_id, v.claim_id, v.fragment_id,
                v.stance.value, v.strength, v.reasoning,
                v.fragment_dated_at, v.processed_at, v.model_signature,
                Jsonb(v.signals) if v.signals else None,
            )
            for v in votes
        ]

        subject_max_dated: dict[str, datetime] = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))
        for v in votes:
            if v.fragment_dated_at > subject_max_dated[v.subject_id]:
                subject_max_dated[v.subject_id] = v.fragment_dated_at

        with self._pg.conn() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO votes (vote_id, subject_id, claim_id, fragment_id, stance,
                        strength, reasoning, fragment_dated_at, processed_at, model_signature, signals)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (subject_id, fragment_id, stance) WHERE invalidated_at IS NULL DO NOTHING
                    """,
                    rows,
                )
                for subject_id, max_dated in subject_max_dated.items():
                    cur.execute(
                        """
                        UPDATE subjects
                        SET last_activity_at = GREATEST(last_activity_at, %s)
                        WHERE subject_id = %s
                        """,
                        (max_dated, subject_id),
                    )

    def fetch_votes_for_subject(
        self,
        subject_id: str,
        as_of: datetime | None = None,
        include_invalidated: bool = False,
    ) -> list[Vote]:
        """Fetch all votes for a subject, optionally filtered to fragment_dated_at <= as_of.

        Used by the claim regenerator (to read the recent vote stream) and
        by traction queries.
        """
        conditions = ["subject_id = %s"]
        params: list = [subject_id]

        if as_of is not None:
            conditions.append("fragment_dated_at <= %s")
            params.append(as_of)

        if not include_invalidated:
            conditions.append("invalidated_at IS NULL")

        sql = (
            f"SELECT * FROM votes WHERE {' AND '.join(conditions)}"
            " ORDER BY fragment_dated_at"
        )
        rows = self._pg.fetch_rows(sql, tuple(params))
        return [self._row_to_vote(r) for r in rows]

    def vote_count_since(self, subject_id: str, since_count: int) -> int:
        """Return number of active votes for subject MINUS since_count.

        Used by the claim regenerator's trigger:
            if vote_count_since(subject, claim.regenerated_at_vote_count)
                >= CLAIM_REGEN_VOTE_GAP: regenerate
        """
        rows = self._pg.fetch_rows(
            "SELECT COUNT(*) AS cnt FROM votes WHERE subject_id = %s AND invalidated_at IS NULL",
            (subject_id,),
        )
        total = int(rows[0]["cnt"]) if rows else 0
        return max(0, total - since_count)

    def invalidate_votes_for_fragment(self, fragment_id: str, reason: str) -> int:
        """Soft-delete all active votes for a fragment (e.g., when a fragment is edited).

        Sets invalidated_at=now() and invalidation_reason=reason. Never hard-deletes.
        Returns the number of rows invalidated.
        """
        return self._pg.execute(
            """
            UPDATE votes
            SET invalidated_at = now(), invalidation_reason = %s
            WHERE fragment_id = %s AND invalidated_at IS NULL
            """,
            (reason, fragment_id),
        )

    # ── fragment_processing ──────────────────────────────────────────────

    def record_processing(
        self,
        fragment_id: str,
        model_signature: str,
        vote_count: int,
        status: ProcessingStatus = ProcessingStatus.SUCCESS,
        error_detail: str | None = None,
    ) -> FragmentProcessing:
        """Insert a fragment_processing row. Zero votes is a valid success."""
        fp = FragmentProcessing(
            fragment_id=fragment_id,
            model_signature=model_signature,
            vote_count=vote_count,
            status=status,
            error_detail=error_detail,
        )
        self._pg.execute(
            """
            INSERT INTO fragment_processing
                (processing_id, fragment_id, processed_at, model_signature, vote_count, status, error_detail)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                fp.processing_id, fp.fragment_id, fp.processed_at,
                fp.model_signature, fp.vote_count, fp.status.value, fp.error_detail,
            ),
        )
        return fp

    def fetch_unprocessed_fragment_ids(
        self,
        model_signature: str,
        after: datetime | None = None,
        limit: int = 500,
    ) -> list[str]:
        """Return fragment_ids that have no SUCCESS row in fragment_processing
        for the given model_signature.

        After replaces the old `load_unprocessed_fragments` semantics (which
        used insight_fragments junction). The new pipeline tracks processing
        per (fragment, model_signature) so re-running with a newer prompt
        version naturally re-processes everything.
        """
        rows = self._pg.fetch_rows(
            """
            SELECT f.fragment_id
            FROM fragments f
            WHERE NOT EXISTS (
                SELECT 1 FROM fragment_processing fp
                WHERE fp.fragment_id    = f.fragment_id
                  AND fp.model_signature = %s
                  AND fp.status          = 'success'
            )
            AND (%s::timestamptz IS NULL OR f.timestamp > %s::timestamptz)
            ORDER BY f.timestamp
            LIMIT %s
            """,
            (model_signature, after, after, limit),
        )
        return [r["fragment_id"] for r in rows]

    def fetch_unprocessed_fragments(
        self,
        model_signature: str,
        limit: int = 500,
    ) -> list:
        """Return Fragment objects that have no SUCCESS row in fragment_processing
        for the given model_signature.

        Combines the ID lookup and the fragment fetch into one query so the
        caller (make_claim_reflect_node) never does N+1 per-fragment fetches.
        Tags and embeddings are populated from the fragments table.
        """
        sql = """
            SELECT f.fragment_id, f.session_id, f.content, f.tags,
                   f.embedding, f.timestamp,
                   COALESCE(
                       array_agg(fe.exchange_id) FILTER (WHERE fe.exchange_id IS NOT NULL),
                       ARRAY[]::text[]
                   ) AS exchange_ids
            FROM fragments f
            LEFT JOIN fragment_exchanges fe ON fe.fragment_id = f.fragment_id
            WHERE NOT EXISTS (
                SELECT 1 FROM fragment_processing fp
                WHERE fp.fragment_id    = f.fragment_id
                  AND fp.model_signature = %s
                  AND fp.status          = 'success'
            )
            GROUP BY f.fragment_id, f.session_id, f.content, f.tags, f.embedding, f.timestamp
            ORDER BY f.timestamp
            LIMIT %s
        """
        return self._pg.fetch_fragments(sql, (model_signature, limit))

    def embed_text(self, text: str):
        """Embed a text string using the repo's embedder. Returns a numpy array."""
        return self._embedder.embed(text)
