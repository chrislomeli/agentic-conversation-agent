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


