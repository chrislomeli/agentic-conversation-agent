# Getting Started — Journal Agent

## What Is This?

Journal Agent is a conversational journaling application. It works like a natural chat: you write freely, and the AI responds as a thoughtful companion — reflecting on what you share, surfacing past memories, and tracking themes and beliefs over time.

Under the hood it uses a LangGraph conversation pipeline, pgvector for semantic search, and a Phase 11 claim-based insight engine that builds a living model of your recurring ideas.

---

## Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| Python | ≥ 3.11 | Managed via `.python-version` |
| [uv](https://docs.astral.sh/uv/) | latest | Package + venv manager |
| Node.js | ≥ 18 | For the React frontend |
| PostgreSQL + pgvector | ≥ 15 | Local or Docker |

### Environment variables

Create a `.env` file (or point `AI_ENV_FILE` at one):

```dotenv
# Required
POSTGRES_URL=postgresql://localhost:5432/journal

# LLM credentials — set whichever provider you use
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional — Ollama local dev (default http://localhost:11434)
OLLAMA_BASE_URL=http://localhost:11434

# Tell pydantic-settings where the file lives (if not in the working dir)
AI_ENV_FILE=/path/to/.env
```

### LLM model selection

Edit `journal_agent/configure/settings.py` → `LLM_ROLE_CONFIG` to choose which model handles each role:

```python
LLM_ROLE_CONFIG: dict[str, LLMLabel] = {
    "conversation": LLMLabel.OLLAMA_LLAMA3,   # main chat model
    "classifier":   LLMLabel.GPT,             # intent / profile classifier
    "extractor":    LLMLabel.GPT,             # end-of-session thread extractor
}
```

Available labels: `GPT`, `GPT_MINI`, `CLAUDE_SONNET`, `CLAUDE_OPUS`, `HAIKU`, `OLLAMA_LLAMA3`.

---

## Database setup

Run the schema once against a fresh database:

```bash
psql $POSTGRES_URL -f sql/schema.sql
```

This creates all tables (`sessions`, `exchanges`, `fragments`, `captures`, `subjects`, `claims`, `votes`, etc.) and the pgvector HNSW indexes.

---

## Backend (FastAPI)

```bash
# Install dependencies
uv sync

# Start the API server (default: http://localhost:8000)
uv run uvicorn journal_agent.api.main:app --reload
```

The API exposes a single streaming endpoint:

```
POST /chat
Content-Type: application/json

{
  "session_id": "my-session",
  "message": "Hello, what's on my mind today?"
}
```

Responses are streamed as Server-Sent Events (SSE).

---

## Frontend (React + Vite)

```bash
cd journal_chat_app

# Install Node dependencies (first time only)
npm install

# Start the dev server (default: http://localhost:5173)
npm run dev
```

The frontend connects to the FastAPI backend at `http://localhost:8000`. Open your browser at `http://localhost:5173` and start chatting.

---

## Console runner (no frontend needed)

For quick local testing without the UI:

```bash
uv run python -m journal_agent.main
```

This drops you into an interactive terminal session.

---

## Commands

Type these directly in either the chat UI or the console:

| Command | Description |
|---|---|
| `/reflect` | Run the Phase 10 cluster-based reflection pipeline and ask the AI to narrate patterns from your recent journal entries. |
| `/reflect2` | Run the Phase 11 claim-based reflection pipeline and ask the AI to narrate your tracked beliefs and their traction. |
| `/recall [topic]` | Semantic search over your journal history. The AI narrates the matching entries. |
| `/save [n] <topic>` | Save the last *n* exchanges from this session as a named note in the captures store. `n` defaults to 1 if omitted. |
| `/save <topic> <text>` | Save inline text directly as a capture (no transcript lookup). |
| `/capture [topic]` | Semantic search over your saved captures (notes created with `/save`). The AI narrates the results. |
| `/quit` | End the session and run the end-of-session pipeline (thread extraction, embedding, archiving). |

Plain text with no `/` prefix is a normal conversation turn.

---

## Running tests

```bash
uv run pytest
```

Integration tests that require a live Postgres instance are tagged `integration` and skipped by default:

```bash
# Run everything including integration tests
uv run pytest -m integration
```

---

## Project layout

```
journal_agent/
  api/          FastAPI app entrypoint
  comms/        Command parsing (parse_user_input)
  configure/    Settings, prompts, context builder, LLM registry
  graph/        LangGraph nodes, routing, state definitions
    nodes/      Individual node factories (classifiers, insight nodes, etc.)
  model/        Pydantic domain models (session, insights, subjects/claims)
  stores/       Postgres repositories (fragments, captures, subjects, etc.)
journal_chat_app/   React + Vite frontend
sql/
  schema.sql    Full database schema — run once to initialise
docs/           Developer documentation (you are here)
```
