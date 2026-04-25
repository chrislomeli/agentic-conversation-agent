# Journal Agent API — Build Plan

## Context

This plan covers the build-out of the FastAPI layer for the journal agent. The
current state: a fully working CLI/LangGraph app with a thin FastAPI scaffold
(`api/main.py` with a fake SSE stream). The goal is a production-grade API that
the next phase (a UI) can sit on top of.

The architecture was designed under a "senior AI architect" lens — LangGraph is
the chosen framework, but used deliberately rather than reflexively. Concretely:

- Per-request graph invocation. **No `interrupt()` ever.**
- Checkpointer for in-flight state, repositories for the canonical archive.
- Conversation graph and end-of-session graph as two compiled graphs.
- Side effects (terminal I/O) pulled out of nodes; the runner does I/O.
- Streaming via LangGraph's `astream_events(version="v2")` — anticipating UI
  growth (status indicators, possible tool calling).

---

## Design Matrix — Active Decisions / Build Tasks

| # | Decision or task | Severity | Effort / Risk | Model | What it means concretely |
|---|---|---|---|---|---|
| **9a** | Decouple AI output from the terminal *(done)* | High | **M / M** | Sonnet (Opus reduces risk on async edge cases) | `get_ai_response` streams via `llm.astream`; terminal client consumes `astream_events(v2)` and prints chunks filtered by `langgraph_node`. No more `talk_to_human` inside the node. |
| **9c** | Make the conversation graph one-shot per turn | High | **M / M** | Sonnet (Opus reduces risk on topology refactor) | Remove `get_user_input` from the graph. Conversation graph starts at `intent_classifier` and ends after `get_ai_response`. The terminal client loops in Python: read stdin → invoke graph for one turn → consume token events → repeat. The API endpoint is the same shape: read HTTP body → invoke graph → SSE the events. Recursion limit becomes irrelevant once each invocation is one turn. Removes the `recursion_limit=1000` band-aid in `main.py`. |
| **1** | EOS pipeline: collapse to one node, linear inside *(done)* | Medium-High | **S / S** | Sonnet | `graph/nodes/eos_pipeline.py::make_end_of_session_node` runs 7 phases sequentially via `_call`. State threaded forward with `model_copy`; LangGraph reducers applied once on the final accumulated dict. `build_end_of_session_graph` is now 1 node + 2 edges. `node_trace` fires per-phase so observability is unchanged. |
| **4** | Adopt checkpointer for in-flight state; repositories canonical archive *(done)* | Medium | **S / S** | Sonnet | `AsyncPostgresSaver` with `thread_id = session_id`. Live `JournalState` between requests; repositories remain the queryable archive. Cleanup job for closed sessions still pending. |
| **3** | Keep classifier-router topology — deliberate, not default | Medium | **XS / XS now, M / M later** | N/A | No work today. Note as "kept pending tool-calling comparison." Latent risk: if a future eval shows tool-calling beats the classifier, refactor touches the conversation graph spine. |
| **5** | Streaming API: `astream_events(version="v2")` *(done)* | Low | **S / S** | Sonnet | `api/streaming.py::graph_stream` consumes `astream_events(v2)`, filters `on_chat_model_stream` by `langgraph_node`, reads `system_message` from checkpointer state after stream. `api/main.py` lifespan builds graphs once; `POST /sessions`, `POST /chat/{id}`, `DELETE /sessions/{id}` are the three session lifecycle endpoints. |
| **7-design** | Make the system evaluable | (supports Critical hardening) | **S / XS** | Sonnet | Audit prompts and classifiers for structured outputs. Confirm prompt versioning. Confirm stable IDs. Patch gaps. Additive, low blast radius. |
| **8-design** | Make the system observable | (supports Critical hardening) | **M / XS** | Sonnet | Structured log fields, decision-point logs, operation-boundary entry/exit. Effort medium because of file count, not per-change difficulty. |

---

## Hardening Matrix — Sequenced After Core API Works

| # | Item | Severity | Trigger to start |
|---|---|---|---|
| **7-hardening** | Eval harness for classifier / decomposer / extractor / fragment_extractor | Critical | Before any prompt or model swap. Before scaling user count. |
| **8-hardening** | Cost/latency telemetry: per-role token counts, per-node latency, retry/failure metrics | Critical | Before the API serves real traffic. |
| **6** | Diversify observability: structured logs + metrics alongside LangSmith | Medium | Same time as #8-hardening — built on the same logging substrate. |

---

## Items Closed Out (no action)

- **#2** — `save_*` nodes: collapses into #1 automatically.
- **#9b** — DB writes inside nodes: idiomatic LangGraph, leave as is.
- **#10** — over-decomposition: judgment call, no specific node flagged.

## Known Issues (tabled)

- **`No human and ai content found for exchange`** — fires from
  `inflate_threads` during the EOS pipeline. Two suspects: (1) the
  `/reflect` and `/recall` paths produce exchanges without a paired
  `on_human_turn`; (2) checkpointer roundtrip of nested `Turn` fields
  inside `Exchange` not fully reconstituting as Pydantic instances.
  Non-blocking — pipeline still finishes. Investigate when convenient.

---

## Suggested Sequencing

The matrix is ordered by severity, not build order. The build sequence we've
been following:

1. ~~**#4** — checkpointer wired (XS warm-up, clarifies the state model).~~ *Done.*
2. ~~**#9a** — decouple AI output from terminal.~~ *Done.*
3. ~~**#9c** — finish the decoupling: remove `get_user_input` from the graph
   and run one turn per invocation. Removes the recursion-limit band-aid.
   Prerequisite for a real API endpoint.~~ *Done.*
4. ~~**#5** — wire the conversation graph into FastAPI as an SSE consumer of
   `astream_events(v2)`. Mostly mechanical once #9c lands.~~ *Done.*
5. ~~**#1** — EOS collapse (independent of the conversation work; can interleave).~~ *Done.*
6. **#7-design** — evaluability audit (additive, low blast radius).
7. **#8-design** — observability slog (benefits from a stable codebase).
8. **Hardening matrix** sequenced after the API is functioning end-to-end.

---

## Why These Decisions (Summary)

- **No interrupts**: couples graph execution to HTTP request lifecycle awkwardly.
  Per-request invocation gives clean HTTP semantics. Per-request invocation
  also requires the conversation graph to be one-shot (no `get_user_input`
  node looping internally) — captured as #9c.
- **Checkpointer kept**: simpler than building incremental session persistence.
  No "two sources of truth" because checkpointer holds in-flight state and
  repositories hold the archive — different lifecycle stages.
- **EOS as one node**: the pipeline is linear ETL with no branching. A graph of
  7 nodes pretends it's a routing problem. One node honestly represents one
  unit of work.
- **`astream_events(v2)`**: project trajectory points to UI growth and likely
  richer event surfaces (status indicators, tool calling). The migration cost
  from `messages` mode to `v2` is real; better to start where you'll land.
- **Classifier-router kept for now**: defensible pattern with cost/latency
  advantages over tool-calling. Worth a future eval comparison; not a current
  rewrite.
