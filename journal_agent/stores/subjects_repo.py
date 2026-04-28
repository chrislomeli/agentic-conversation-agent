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

import logging
from datetime import datetime

from journal_agent.model.insights import (
    Claim,
    FragmentProcessing,
    ProcessingStatus,
    ProposedSubject,
    Stance,
    Subject,
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
            INSERT INTO votes (..., claim_version_at_vote=1)

        Returns the persisted records (with server-assigned IDs and timestamps).
        """
        raise NotImplementedError("create_subject_with_claim — skeleton")

    def get_subject(self, subject_id: str) -> Subject | None:
        """Fetch a single Subject by id. Returns None on miss."""
        raise NotImplementedError("get_subject — skeleton")

    def list_active_subjects(self, limit: int = 100) -> list[Subject]:
        """List subjects with status='active', most-recently-active first.

        Used by the subject_proposer to brief the LLM on existing subjects.
        """
        raise NotImplementedError("list_active_subjects — skeleton")

    def mark_subject_status(self, subject_id: str, status: str) -> None:
        """Set subjects.status (e.g., dormant, superseded, merged).

        Status enum values: 'active' | 'dormant' | 'superseded' | 'merged'.
        """
        raise NotImplementedError("mark_subject_status — skeleton")

    # ── claims ───────────────────────────────────────────────────────────

    def get_current_claim(self, subject_id: str) -> Claim | None:
        """Fetch the row in `claims` where subject_id matches and is_current=TRUE."""
        raise NotImplementedError("get_current_claim — skeleton")

    def regenerate_claim(
        self,
        subject_id: str,
        new_text: str,
        regenerated_at_vote_count: int,
    ) -> Claim:
        """Append a new Claim version and flip is_current.

        Real SQL (TODO):
            UPDATE claims SET is_current=FALSE WHERE subject_id=...
                AND is_current=TRUE
            INSERT INTO claims (..., version=N+1, is_current=TRUE,
                regenerated_at_vote_count=...)
                with embedding over `label || new_text`

        Returns the new Claim. Old versions are retained for audit.
        """
        raise NotImplementedError("regenerate_claim — skeleton")

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
        raise NotImplementedError("search_candidate_subjects — skeleton")

    # ── votes ────────────────────────────────────────────────────────────

    def insert_votes(self, votes: list[Vote]) -> None:
        """Bulk-insert votes; updates each subject's last_activity_at to max(
        current, vote.fragment_dated_at).

        No-op on empty list. Idempotent against the
        (subject_id, fragment_id, stance) unique-active constraint —
        duplicates are silently dropped via ON CONFLICT DO NOTHING.
        """
        raise NotImplementedError("insert_votes — skeleton")

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
        raise NotImplementedError("fetch_votes_for_subject — skeleton")

    def vote_count_since(self, subject_id: str, since_count: int) -> int:
        """Return number of active votes for subject MINUS since_count.

        Used by the claim regenerator's trigger:
            if vote_count_since(subject, claim.regenerated_at_vote_count)
                >= CLAIM_REGEN_VOTE_GAP: regenerate
        """
        raise NotImplementedError("vote_count_since — skeleton")

    def invalidate_votes_for_fragment(self, fragment_id: str, reason: str) -> int:
        """Soft-delete all active votes for a fragment (e.g., when a fragment is edited).

        Sets invalidated_at=now() and invalidation_reason=reason. Never hard-deletes.
        Returns the number of rows invalidated.
        """
        raise NotImplementedError("invalidate_votes_for_fragment — skeleton")

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
        raise NotImplementedError("record_processing — skeleton")

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
        raise NotImplementedError("fetch_unprocessed_fragment_ids — skeleton")
