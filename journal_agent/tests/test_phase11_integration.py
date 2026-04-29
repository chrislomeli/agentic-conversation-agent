"""test_phase11_integration.py — Integration tests for SubjectsRepository.

Requires a running Postgres instance. All Phase 11 tables are created in the
'test_phase11' Postgres schema at session start and dropped on teardown, so
production data in the 'public' schema is never touched.

Run only these tests:
    pytest -m integration -v

Skip these tests (fast CI):
    pytest -m 'not integration'

The POSTGRES_URL env var is read from the environment (or .env via the app
settings). Override for a different target:
    POSTGRES_URL=postgresql://localhost/mydb pytest -m integration
"""

from __future__ import annotations

import os
from datetime import datetime

import psycopg
import pytest
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from journal_agent.model.insights import InitialVote, ProcessingStatus, ProposedSubject, Stance, Vote

TEST_SCHEMA = "test_phase11"

# ── DDL for the test schema ───────────────────────────────────────────────────
# FK constraints are omitted; we are testing repo logic, not DB referential
# integrity. The partial unique index on votes IS included because insert_votes
# has an ON CONFLICT clause that targets it.

_DDL = f"""
CREATE TABLE IF NOT EXISTS {TEST_SCHEMA}.fragments (
    fragment_id  TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    content      TEXT NOT NULL,
    tags         JSONB,
    embedding    vector(384),
    timestamp    TIMESTAMPTZ NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {TEST_SCHEMA}.subjects (
    subject_id        TEXT PRIMARY KEY,
    label             TEXT NOT NULL,
    description       TEXT,
    status            TEXT NOT NULL DEFAULT 'active',
    parent_subject_id TEXT,
    merged_into_id    TEXT,
    created_at        TIMESTAMPTZ DEFAULT now(),
    last_activity_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS {TEST_SCHEMA}.claims (
    claim_id                  TEXT PRIMARY KEY,
    subject_id                TEXT NOT NULL,
    text                      TEXT NOT NULL,
    version                   INT  NOT NULL,
    is_current                BOOLEAN NOT NULL DEFAULT FALSE,
    embedding                 vector(384),
    regenerated_at_vote_count INT NOT NULL DEFAULT 0,
    created_at                TIMESTAMPTZ DEFAULT now(),
    UNIQUE (subject_id, version)
);

CREATE UNIQUE INDEX IF NOT EXISTS test11_claims_one_current_idx
    ON {TEST_SCHEMA}.claims (subject_id) WHERE is_current = TRUE;

CREATE TABLE IF NOT EXISTS {TEST_SCHEMA}.votes (
    vote_id             TEXT PRIMARY KEY,
    subject_id          TEXT NOT NULL,
    claim_id            TEXT NOT NULL,
    fragment_id         TEXT NOT NULL,
    stance              TEXT NOT NULL,
    strength            DOUBLE PRECISION NOT NULL,
    reasoning           TEXT NOT NULL,
    fragment_dated_at   TIMESTAMPTZ NOT NULL,
    processed_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_signature     TEXT NOT NULL,
    signals             JSONB,
    invalidated_at      TIMESTAMPTZ,
    invalidation_reason TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS test11_votes_unique_active_idx
    ON {TEST_SCHEMA}.votes (subject_id, fragment_id, stance) WHERE invalidated_at IS NULL;

CREATE TABLE IF NOT EXISTS {TEST_SCHEMA}.fragment_processing (
    processing_id    TEXT PRIMARY KEY,
    fragment_id      TEXT NOT NULL,
    processed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_signature  TEXT NOT NULL,
    vote_count       INT  NOT NULL DEFAULT 0,
    status           TEXT NOT NULL,
    error_detail     TEXT
);
"""

_TRUNCATE = (
    f"TRUNCATE {TEST_SCHEMA}.fragment_processing, {TEST_SCHEMA}.votes, "
    f"{TEST_SCHEMA}.claims, {TEST_SCHEMA}.subjects, {TEST_SCHEMA}.fragments"
)

# ── Session-scoped fixtures ───────────────────────────────────────────────────


@pytest.fixture(scope="session")
def pg_url() -> str:
    from journal_agent.configure.settings import get_settings
    return os.environ.get("POSTGRES_URL", get_settings().postgres_url)


@pytest.fixture(scope="session")
def schema_setup(pg_url: str):
    """Create test schema + tables once per session. Drop everything on teardown."""
    with psycopg.connect(pg_url, autocommit=True) as conn:
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {TEST_SCHEMA}")
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        for stmt in _DDL.split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)
    yield
    with psycopg.connect(pg_url, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA} CASCADE")


@pytest.fixture(scope="session")
def pg(schema_setup, pg_url: str):
    """PgGateway backed by a pool whose connections target the test schema."""
    from journal_agent.stores.pg_gateway import PgGateway

    def _configure(conn):
        register_vector(conn)
        conn.execute(f"SET search_path TO {TEST_SCHEMA}, public")
        conn.commit()

    gw = PgGateway.__new__(PgGateway)
    gw._pool = ConnectionPool(
        conninfo=pg_url,
        min_size=1,
        max_size=3,
        kwargs={"row_factory": dict_row},
        open=False,
        configure=_configure,
    )
    gw._pool.open(wait=True)
    yield gw
    gw._pool.close()


# ── Function-scoped fixtures ──────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clean(pg):
    """Truncate all test tables before every test."""
    pg.execute(_TRUNCATE)
    yield


@pytest.fixture
def repo(pg):
    from journal_agent.stores.subjects_repo import SubjectsRepository
    return SubjectsRepository(pg_gateway=pg)


# ── Builders ──────────────────────────────────────────────────────────────────


def _proposed(label: str = "stance on solitude") -> ProposedSubject:
    return ProposedSubject(
        label=label,
        description="Tracks the user's relationship with alone time.",
        initial_claim="User actively seeks and values solitude.",
        initial_vote=InitialVote(stance=Stance.SUPPORT, strength=0.9, reasoning="Stated directly."),
    )


def _first_vote(fragment_id: str = "frag-001") -> Vote:
    return Vote(
        subject_id="placeholder",
        claim_id="placeholder",
        fragment_id=fragment_id,
        stance=Stance.SUPPORT,
        strength=0.8,
        reasoning="User explicitly states they value being alone.",
        fragment_dated_at=datetime(2026, 1, 15, 9, 0),
        model_signature="test/stance_classifier_v1",
    )


def _seed_fragment(pg, fragment_id: str = "frag-001", ts: datetime | None = None):
    pg.execute(
        "INSERT INTO fragments (fragment_id, session_id, content, timestamp) VALUES (%s, %s, %s, %s)",
        (fragment_id, "sess-001", "some content", ts or datetime(2026, 1, 15)),
    )


# ── Tests: subjects ───────────────────────────────────────────────────────────


@pytest.mark.integration
def test_create_subject_with_claim_persists_all_three_rows(repo, pg):
    subject, claim, vote = repo.create_subject_with_claim(_proposed(), _first_vote())

    s_rows = pg.fetch_rows("SELECT * FROM subjects WHERE subject_id = %s", (subject.subject_id,))
    c_rows = pg.fetch_rows("SELECT * FROM claims WHERE claim_id = %s", (claim.claim_id,))
    v_rows = pg.fetch_rows("SELECT * FROM votes WHERE vote_id = %s", (vote.vote_id,))

    assert len(s_rows) == 1
    assert s_rows[0]["label"] == "stance on solitude"
    assert len(c_rows) == 1
    assert c_rows[0]["version"] == 1
    assert c_rows[0]["is_current"] is True
    assert len(v_rows) == 1
    assert v_rows[0]["subject_id"] == subject.subject_id
    assert v_rows[0]["claim_id"] == claim.claim_id


@pytest.mark.integration
def test_get_subject_hit_and_miss(repo):
    subject, _, _ = repo.create_subject_with_claim(_proposed(), _first_vote())

    found = repo.get_subject(subject.subject_id)
    assert found is not None
    assert found.label == "stance on solitude"

    assert repo.get_subject("does-not-exist") is None


@pytest.mark.integration
def test_list_active_subjects_filters_by_status(repo):
    repo.create_subject_with_claim(_proposed("topic A"), _first_vote("frag-001"))
    s2, _, _ = repo.create_subject_with_claim(_proposed("topic B"), _first_vote("frag-002"))
    repo.create_subject_with_claim(_proposed("topic C"), _first_vote("frag-003"))

    repo.mark_subject_status(s2.subject_id, "dormant")

    active = repo.list_active_subjects()
    labels = {s.label for s in active}
    assert "topic A" in labels
    assert "topic C" in labels
    assert "topic B" not in labels


@pytest.mark.integration
def test_mark_subject_status(repo):
    subject, _, _ = repo.create_subject_with_claim(_proposed(), _first_vote())
    repo.mark_subject_status(subject.subject_id, "dormant")

    found = repo.get_subject(subject.subject_id)
    assert found.status.value == "dormant"


# ── Tests: claims ─────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_get_current_claim(repo):
    subject, claim, _ = repo.create_subject_with_claim(_proposed(), _first_vote())

    current = repo.get_current_claim(subject.subject_id)
    assert current is not None
    assert current.claim_id == claim.claim_id
    assert current.is_current is True
    assert current.version == 1


@pytest.mark.integration
def test_regenerate_claim_increments_version_and_flips_current(repo):
    subject, old_claim, _ = repo.create_subject_with_claim(_proposed(), _first_vote())

    new_claim = repo.regenerate_claim(
        subject_id=subject.subject_id,
        new_text="User has deepened their practice and now schedules daily solitude.",
        regenerated_at_vote_count=5,
    )

    assert new_claim.version == 2
    assert new_claim.is_current is True
    assert new_claim.regenerated_at_vote_count == 5

    stale = repo.get_current_claim(subject.subject_id)
    assert stale.claim_id == new_claim.claim_id

    third = repo.regenerate_claim(
        subject_id=subject.subject_id,
        new_text="Solitude is now a non-negotiable daily anchor.",
        regenerated_at_vote_count=12,
    )
    assert third.version == 3


@pytest.mark.integration
def test_search_candidate_subjects_returns_nearest(repo):
    repo.create_subject_with_claim(
        _proposed("meditation and Buddhism"),
        _first_vote("frag-001"),
    )
    repo.create_subject_with_claim(
        ProposedSubject(
            label="dog walking and outdoors",
            description="Tracks relationship with pets and outdoor activity.",
            initial_claim="User walks the dog daily and enjoys it.",
            initial_vote=InitialVote(stance=Stance.SUPPORT, strength=0.8, reasoning="..."),
        ),
        _first_vote("frag-002"),
    )

    from journal_agent.stores.embedder import Embedder
    embedder = Embedder()
    query_vec = embedder.embed("mindfulness meditation practice").tolist()

    results = repo.search_candidate_subjects(query_embedding=query_vec, top_k=5, min_similarity=0.0)

    assert len(results) == 2
    labels = [s.label for s, _, _ in results]
    assert labels[0] == "meditation and Buddhism"


# ── Tests: votes ──────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_insert_votes_bulk_and_idempotent(repo, pg):
    subject, claim, _ = repo.create_subject_with_claim(_proposed(), _first_vote())

    extra_vote = Vote(
        subject_id=subject.subject_id,
        claim_id=claim.claim_id,
        fragment_id="frag-002",
        stance=Stance.SUPPORT,
        strength=0.7,
        reasoning="Another entry confirming the pattern.",
        fragment_dated_at=datetime(2026, 2, 1),
        model_signature="test/stance_classifier_v1",
    )
    repo.insert_votes([extra_vote])

    rows = pg.fetch_rows("SELECT * FROM votes WHERE subject_id = %s", (subject.subject_id,))
    assert len(rows) == 2

    repo.insert_votes([extra_vote])
    rows2 = pg.fetch_rows("SELECT * FROM votes WHERE subject_id = %s", (subject.subject_id,))
    assert len(rows2) == 2


@pytest.mark.integration
def test_fetch_votes_for_subject_with_as_of(repo):
    subject, claim, early_vote = repo.create_subject_with_claim(_proposed(), _first_vote())

    late_vote = Vote(
        subject_id=subject.subject_id,
        claim_id=claim.claim_id,
        fragment_id="frag-002",
        stance=Stance.SUPPORT,
        strength=0.6,
        reasoning="Later entry.",
        fragment_dated_at=datetime(2026, 6, 1),
        model_signature="test/stance_classifier_v1",
    )
    repo.insert_votes([late_vote])

    all_votes = repo.fetch_votes_for_subject(subject.subject_id)
    assert len(all_votes) == 2

    early_only = repo.fetch_votes_for_subject(subject.subject_id, as_of=datetime(2026, 3, 1))
    assert len(early_only) == 1
    assert early_only[0].fragment_id == early_vote.fragment_id


@pytest.mark.integration
def test_invalidate_and_include_invalidated(repo):
    subject, claim, _ = repo.create_subject_with_claim(_proposed(), _first_vote("frag-001"))

    count = repo.invalidate_votes_for_fragment("frag-001", reason="fragment edited")
    assert count == 1

    active = repo.fetch_votes_for_subject(subject.subject_id)
    assert len(active) == 0

    with_invalid = repo.fetch_votes_for_subject(subject.subject_id, include_invalidated=True)
    assert len(with_invalid) == 1
    assert with_invalid[0].invalidation_reason == "fragment edited"


@pytest.mark.integration
def test_vote_count_since(repo):
    subject, claim, _ = repo.create_subject_with_claim(_proposed(), _first_vote())

    extra = Vote(
        subject_id=subject.subject_id,
        claim_id=claim.claim_id,
        fragment_id="frag-002",
        stance=Stance.SUPPORT,
        strength=0.7,
        reasoning="...",
        fragment_dated_at=datetime(2026, 2, 1),
        model_signature="test/v1",
    )
    repo.insert_votes([extra])

    assert repo.vote_count_since(subject.subject_id, since_count=0) == 2
    assert repo.vote_count_since(subject.subject_id, since_count=1) == 1
    assert repo.vote_count_since(subject.subject_id, since_count=2) == 0
    assert repo.vote_count_since(subject.subject_id, since_count=99) == 0


# ── Tests: fragment_processing ────────────────────────────────────────────────


@pytest.mark.integration
def test_record_processing(repo, pg):
    fp = repo.record_processing(
        fragment_id="frag-001",
        model_signature="test/v1",
        vote_count=3,
        status=ProcessingStatus.SUCCESS,
    )
    rows = pg.fetch_rows("SELECT * FROM fragment_processing WHERE processing_id = %s", (fp.processing_id,))
    assert len(rows) == 1
    assert rows[0]["vote_count"] == 3
    assert rows[0]["status"] == "success"


@pytest.mark.integration
def test_fetch_unprocessed_fragment_ids(repo, pg):
    _seed_fragment(pg, "frag-001", datetime(2026, 1, 10))
    _seed_fragment(pg, "frag-002", datetime(2026, 1, 20))
    _seed_fragment(pg, "frag-003", datetime(2026, 1, 30))

    repo.record_processing("frag-002", model_signature="test/v1", vote_count=1)

    unprocessed = repo.fetch_unprocessed_fragment_ids(model_signature="test/v1")
    assert set(unprocessed) == {"frag-001", "frag-003"}

    after_filter = repo.fetch_unprocessed_fragment_ids(
        model_signature="test/v1",
        after=datetime(2026, 1, 15),
    )
    assert after_filter == ["frag-003"]
