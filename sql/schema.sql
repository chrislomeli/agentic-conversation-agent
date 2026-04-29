-- schema.sql — Postgres + pgvector schema for the journal agent.
--
-- Run once against a fresh database:
--     psql $POSTGRES_URL -f data/schema.sql
--
-- Design notes:
--   * "threads" and "classified_threads" collapse into one table; tags are
--     NULL until the classifier runs, then populated via upsert.
--   * exchange_ids arrays on threads and fragments are exploded into
--     junction tables (thread_exchanges, fragment_exchanges).
--   * user_profiles is single-row, keyed by a default 'default' user_id —
--     no versioning; upgrade to a history table later if needed.
--   * Embedding dimension defaults to 384 to match ChromaDB's default
--     sentence-transformer (all-MiniLM-L6-v2). Change if you swap models.

CREATE EXTENSION IF NOT EXISTS vector;

-- ── sessions (parent) ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    started_at   TIMESTAMPTZ DEFAULT now(),
    ended_at     TIMESTAMPTZ
);

-- ── exchanges (raw human/AI turn pairs) ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS exchanges (
    exchange_id    TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    timestamp      TIMESTAMPTZ NOT NULL,
    human_content  TEXT,
    ai_content     TEXT
);
CREATE INDEX IF NOT EXISTS exchanges_session_ts_idx ON exchanges (session_id, timestamp);

-- ── threads (merged: pre-classification + post-classification) ────────────────
-- Natural key is (session_id, thread_name); synthetic PK is session_id:thread_name.
-- tags starts NULL (from save_threads), gets populated by save_classified_threads.
CREATE TABLE IF NOT EXISTS threads (
    thread_id    TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    thread_name  TEXT NOT NULL,
    tags         JSONB,
    UNIQUE (session_id, thread_name)
);

-- ── thread_exchanges (junction) ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS thread_exchanges (
    thread_id    TEXT NOT NULL REFERENCES threads(thread_id)     ON DELETE CASCADE,
    exchange_id  TEXT NOT NULL REFERENCES exchanges(exchange_id) ON DELETE CASCADE,
    position     INT  NOT NULL,
    PRIMARY KEY (thread_id, exchange_id)
);

-- ── fragments (content + embedding) ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fragments (
    fragment_id   TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    content       TEXT NOT NULL,
    tags          JSONB,
    embedding     vector(384),            -- NULL-able; populated when embedding is available
    timestamp     TIMESTAMPTZ NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT now()
);
-- HNSW index only applies to non-NULL rows, so leaving embedding NULL is safe.
CREATE INDEX IF NOT EXISTS fragments_embedding_idx
    ON fragments USING hnsw (embedding vector_cosine_ops);

-- ── fragment_exchanges (junction) ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fragment_exchanges (
    fragment_id  TEXT NOT NULL REFERENCES fragments(fragment_id) ON DELETE CASCADE,
    exchange_id  TEXT NOT NULL REFERENCES exchanges(exchange_id) ON DELETE CASCADE,
    PRIMARY KEY (fragment_id, exchange_id)
);

-- ── user_profiles (single row, no versioning) ─────────────────────────────────
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id            TEXT PRIMARY KEY,
    response_style     TEXT,
    explanation_depth  TEXT,
    tone               TEXT,
    decision_style     TEXT,
    learning_style     TEXT,
    interests          JSONB,
    pet_peeves         JSONB,
    active_projects    JSONB,
    recurring_themes   JSONB,
    human_label        TEXT,
    ai_label           TEXT,
    updated_at         TIMESTAMPTZ DEFAULT now()
);

-- ── insights ─────────────────────────────────
create table insights
(
    insight_id      text not null
        primary key,
    session_id      text not null
        references sessions
            on delete cascade,
    label           text not null,
    body            text not null,
    verifier_status varchar(20),
    confidence      double precision,
    embedding       vector(384),
    created_at      timestamp with time zone default now()
);

create index insights_fragments_embedding_idx
    on insights using hnsw (embedding vector_cosine_ops);

-- ── insights to fragments relational table ─────────────────────────────────
create table insight_fragments
(
    insight_id  text not null
        references insights
            on delete cascade,
    fragment_id text not null
        references fragments
            on delete cascade,
    primary key (insight_id, fragment_id)
);


-- ═════════════════════════════════════════════════════════════════════════════
-- Phase 11 — Claim-based insights (subject / claim / vote model)
--
-- Coexists with the Phase 10 `insights` / `insight_fragments` tables above.
-- Design doc: design/phase11-claim-based-insights.md
-- ═════════════════════════════════════════════════════════════════════════════

-- ── subjects ──────────────────────────────────────────────────────────────────
-- Stable handles for tracked ideas. label/status/last_activity_at mutate;
-- identity (subject_id) is stable across the subject's lifetime.
CREATE TABLE IF NOT EXISTS subjects (
    subject_id        TEXT PRIMARY KEY,
    label             TEXT NOT NULL,
    description       TEXT,
    status            TEXT NOT NULL DEFAULT 'active',  -- active|dormant|superseded|merged
    parent_subject_id TEXT REFERENCES subjects(subject_id),
    merged_into_id    TEXT REFERENCES subjects(subject_id),
    created_at        TIMESTAMPTZ DEFAULT now(),
    last_activity_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS subjects_status_idx ON subjects (status);
CREATE INDEX IF NOT EXISTS subjects_last_activity_idx ON subjects (last_activity_at);

-- ── claims ────────────────────────────────────────────────────────────────────
-- Versioned phrasing of the user's position on a subject. The LLM regenerates
-- the text as evidence accumulates; old versions are kept for audit. Exactly
-- one row per subject has is_current = true. Embedding is over (label || text)
-- and is used for routing new fragments to candidate subjects.
CREATE TABLE IF NOT EXISTS claims (
    claim_id                  TEXT PRIMARY KEY,
    subject_id                TEXT NOT NULL REFERENCES subjects(subject_id) ON DELETE CASCADE,
    text                      TEXT NOT NULL,
    version                   INT  NOT NULL,
    is_current                BOOLEAN NOT NULL DEFAULT FALSE,
    embedding                 vector(384),  -- NULL until embedded
    regenerated_at_vote_count INT  NOT NULL DEFAULT 0,
    created_at                TIMESTAMPTZ DEFAULT now(),
    UNIQUE (subject_id, version)
);
-- Enforce single current claim per subject via partial unique index.
CREATE UNIQUE INDEX IF NOT EXISTS claims_one_current_per_subject_idx
    ON claims (subject_id) WHERE is_current = TRUE;
CREATE INDEX IF NOT EXISTS claims_embedding_idx
    ON claims USING hnsw (embedding vector_cosine_ops);

-- ── votes ─────────────────────────────────────────────────────────────────────
-- Append-only timestamped evidence. claim_id is the claim that was evaluated
-- against when the vote was cast; it never changes. subject_id is denormalized
-- for efficient traction queries ("all votes for subject X") without a join
-- through claims. fragment_dated_at drives all "as-of" belief queries;
-- processed_at is for audit only.
CREATE TABLE IF NOT EXISTS votes (
    vote_id             TEXT PRIMARY KEY,
    subject_id          TEXT NOT NULL REFERENCES subjects(subject_id) ON DELETE CASCADE,
    claim_id            TEXT NOT NULL REFERENCES claims(claim_id)     ON DELETE CASCADE,
    fragment_id         TEXT NOT NULL REFERENCES fragments(fragment_id) ON DELETE CASCADE,
    stance              TEXT NOT NULL,  -- support|contradict
    strength            DOUBLE PRECISION NOT NULL CHECK (strength >= 0.0 AND strength <= 1.0),
    reasoning           TEXT NOT NULL,
    fragment_dated_at   TIMESTAMPTZ NOT NULL,
    processed_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_signature     TEXT NOT NULL,
    signals             JSONB,
    invalidated_at      TIMESTAMPTZ,
    invalidation_reason TEXT
);
-- One active vote per (subject, fragment, stance). Allows ambivalence
-- (support+contradict on the same subject from one fragment) but not duplicates.
CREATE UNIQUE INDEX IF NOT EXISTS votes_unique_active_idx
    ON votes (subject_id, fragment_id, stance) WHERE invalidated_at IS NULL;
CREATE INDEX IF NOT EXISTS votes_subject_dated_idx
    ON votes (subject_id, fragment_dated_at);
CREATE INDEX IF NOT EXISTS votes_fragment_idx
    ON votes (fragment_id);

-- ── fragment_processing ───────────────────────────────────────────────────────
-- Bookkeeping: "we looked at fragment F with model M at time T". A run that
-- produced zero votes is still a successful processing — distinguishable from
-- "haven't looked yet". Used by the reflect node's loop to skip processed work.
CREATE TABLE IF NOT EXISTS fragment_processing (
    processing_id    TEXT PRIMARY KEY,
    fragment_id      TEXT NOT NULL REFERENCES fragments(fragment_id) ON DELETE CASCADE,
    processed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_signature  TEXT NOT NULL,
    vote_count       INT  NOT NULL DEFAULT 0,
    status           TEXT NOT NULL,  -- success|error|partial
    error_detail     TEXT
);
CREATE INDEX IF NOT EXISTS fragment_processing_fragment_idx
    ON fragment_processing (fragment_id);
CREATE INDEX IF NOT EXISTS fragment_processing_processed_at_idx
    ON fragment_processing (processed_at);

