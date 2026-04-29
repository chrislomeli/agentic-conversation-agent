# LangGraph Architect & Developer Rubric

> A professional-grade checklist for mastering LangGraph. Items marked ✅ are
> demonstrated in the **journal_agent** project. Items marked ⬜ are not yet
> implemented here and represent growth opportunities.

---

## 1 — Graph Fundamentals

| Status | Skill | Evidence / Notes |
|--------|-------|------------------|
| ✅ | **StateGraph with typed Pydantic state** | `JournalState`, `ReflectionState` — full Pydantic `BaseModel` schemas with field docs |
| ✅ | **Append-reducers (`Annotated[..., add]`)** | `session_messages` uses `add_messages`, `transcript`/`threads`/`classified_threads` use `add` |
| ✅ | **Multiple compiled graphs in one app** | Conversation graph, EOS graph, reflection graph, claim-reflection graph |
| ✅ | **START / END wiring** | Every graph wires `START →` entry and `→ END` exits explicitly |
| ✅ | **Linear (sequential) edges** | EOS pipeline is a 7-phase sequential chain inside one super-node |
| ✅ | **Conditional edges with routing functions** | `add_conditional_edges` with `route_on_start`, `route_on_intent`, `route_on_profile`, `goto()`, `should_cold_start` |
| ✅ | **Dynamic path-map conditional edges** | `should_cold_start` returns dict-based path map `{ "cluster_seed_subjects": ..., "route_candidates": ... }` |
| ⬜ | **`Send()` API for map-reduce / fan-out** | Fan-out is done with `asyncio.gather` inside nodes, not via LangGraph's native `Send()` primitive |
| ⬜ | **Subgraph nesting (nested `CompiledGraph` as a node)** | Reflection graphs are invoked manually via `await graph.ainvoke()` inside a wrapper node, not wired as native subgraph nodes |

## 2 — State Design & Data Modeling

| Status | Skill | Evidence / Notes |
|--------|-------|------------------|
| ✅ | **Rich domain model with Pydantic** | `Fragment`, `Exchange`, `ThreadSegment`, `Insight`, `Subject`, `Claim`, `Vote`, `FragmentWorkItem`, etc. |
| ✅ | **Enum-driven control flow** | `StatusValue`, `UserCommandValue`, `Stance`, `SubjectStatus`, `ProcessingStatus` |
| ✅ | **Nested state schemas** | `ContextSpecification`, `WindowParams`, `UserProfile` embedded in graph state |
| ✅ | **Separate state schemas per graph** | `JournalState` for conversation/EOS, `ReflectionState` for reflection graphs |
| ✅ | **Partial-dict return convention** | Every node returns only changed fields; LangGraph merges via reducers |

## 3 — Node Patterns

| Status | Skill | Evidence / Notes |
|--------|-------|------------------|
| ✅ | **Factory functions (`make_*`) with dependency injection** | Every node is a closure: `make_get_ai_response(llm, session_store)`, `make_intent_classifier(llm)`, etc. |
| ✅ | **Sync and async node functions** | `exchange_decomposer` is sync; `thread_classifier`, `get_ai_response` are async |
| ✅ | **Per-node observability decorator** | `@node_trace` logs session_id, elapsed_ms, status per node execution |
| ✅ | **Bounded-concurrency fan-out inside nodes** | `asyncio.Semaphore(max_concurrency)` + `asyncio.gather` for thread_classifier, fragment_extractor, insight nodes |
| ✅ | **Error-to-state propagation** | Every node catches exceptions → returns `{ "status": ERROR, "error_message": ... }` |
| ✅ | **Super-node (multi-phase sequential pipeline)** | `make_end_of_session_node` runs 7 phases inside one graph node with `model_copy(update=...)` state threading |

## 4 — Routing & Control Flow

| Status | Skill | Evidence / Notes |
|--------|-------|------------------|
| ✅ | **Command-based routing (user commands)** | `route_on_start` dispatches `/reflect`, `/recall`, `/save`, `/capture` to dedicated nodes |
| ✅ | **Classification-based routing** | `route_on_intent` uses intent classifier output to pick prompt strategy + retrieval depth |
| ✅ | **Reusable routing helpers** | `goto(node, on_completion)` closure handles error/completion guards uniformly |
| ✅ | **Multi-level conditional routing** | START → intent → profile → retrieve → response (4-level conditional chain) |
| ✅ | **Cold-start routing** | `should_cold_start` counts DB rows to decide between seeding vs. per-fragment path |
| ⬜ | **Cycle / loop inside the graph** | Conversation loop is driven by the Python runner, not a graph cycle (deliberate design choice) |

## 5 — Checkpointing & Persistence

| Status | Skill | Evidence / Notes |
|--------|-------|------------------|
| ✅ | **Postgres-backed async checkpointer** | `AsyncPostgresSaver` with custom `JsonPlusSerializer` + allowed msgpack modules |
| ✅ | **Thread-based session isolation** | `thread_id = session_id` keys conversation state per session |
| ✅ | **Checkpoint state reads (`aget_state`)** | Runner and SSE streaming read `system_message` from checkpointed snapshot after each turn |
| ✅ | **Shared checkpointer across graphs** | Conversation graph and EOS graph share the same checkpointer keyed by `thread_id` |
| ⬜ | **Time-travel / state replay** | No rollback to prior checkpoint versions |
| ⬜ | **`get_state_history` for debugging** | Not used in the codebase |

## 6 — LLM Integration

| Status | Skill | Evidence / Notes |
|--------|-------|------------------|
| ✅ | **Multi-provider LLM abstraction** | `LLMClient` wraps OpenAI, Anthropic, Ollama via `create_llm_client` factory |
| ✅ | **Role-based LLM registry** | `LLMRegistry` maps roles (conversation, classifier, extractor) to different models |
| ✅ | **Structured output (`.with_structured_output`)** | `ScoreCard`, `ThreadSegmentList`, `ThreadClassificationResponse`, `FragmentDraftList`, `StanceResponse`, `ProposerResponse`, `BatchStanceResponse`, `BatchVerifierResponse` |
| ✅ | **Async streaming (`astream`)** | `get_ai_response` node uses `llm.astream(messages)` for token-by-token streaming |
| ✅ | **Token-budget-aware context assembly** | `ContextBuilder` estimates tokens via tiktoken, prunes by priority (retrieved → recent → session) |
| ✅ | **Prompt versioning** | `get_prompt_version(key)` tracks prompt versions; eval records stamp prompt_version |
| ✅ | **Multiple prompt strategies** | Conversation, Socratic, Guidance, plus 15+ classifier/extractor prompts |
| ⬜ | **Tool-calling / function-calling** | LLM is never given tools to call; all "actions" are hard-wired as graph nodes |
| ⬜ | **ReAct (Reasoning + Acting) agent** | No observe→think→act loop; the graph is a fixed DAG, not a ReAct agent |
| ⬜ | **LLM-as-judge pattern** | Verification nodes score outputs, but don't use a separate "judge" model |
| ⬜ | **Few-shot examples in prompts** | Prompts use instruction-style templates, not few-shot exemplars |
| ⬜ | **Fallback / retry chains** | No automatic retry on LLM failure; errors propagate to state |

## 7 — Streaming & Transport

| Status | Skill | Evidence / Notes |
|--------|-------|------------------|
| ✅ | **`astream_events(version="v2")`** | Both terminal runner and FastAPI endpoint consume the v2 event stream |
| ✅ | **Server-Sent Events (SSE) over FastAPI** | `StreamingResponse` with `text/event-stream`, token/system/done/error event types |
| ✅ | **Terminal streaming consumer** | `stream_ai_response_to_terminal` filters `on_chat_model_stream` events for `get_ai_response` node only |
| ✅ | **Node-scoped event filtering** | Both consumers check `metadata.langgraph_node == "get_ai_response"` to avoid leaking classifier LLM events |
| ⬜ | **WebSocket transport** | SSE only; no bidirectional WebSocket channel |
| ⬜ | **LangServe deployment** | Custom FastAPI app, not LangServe |

## 8 — Testing & Evaluation

| Status | Skill | Evidence / Notes |
|--------|-------|------------------|
| ✅ | **Comprehensive unit test suite** | 20+ test files covering nodes, graph wiring, streaming, API, storage, config, observability |
| ✅ | **Mock-based graph testing** | `_make_conversation_mock`, mock LLM clients, monkeypatched node functions |
| ✅ | **Eval framework with JSONL output** | `evals/runner.py` runs classifiers against fixtures, writes `EvalRecord` JSONL for diffing |
| ✅ | **Fixture-driven eval suites** | `evals/fixtures.py` builds typed test states; eval runner chains EOS classifiers in pipeline order |
| ✅ | **Score card tests** | `test_score_card.py` validates intent→specification mapping exhaustively |
| ✅ | **Async test support** | `pytest-asyncio` with `asyncio_mode = "auto"` |
| ✅ | **Integration test markers** | `@pytest.mark.integration` separates tests requiring live Postgres |
| ⬜ | **End-to-end graph integration test** | Tests exercise nodes individually or via mocked graphs, not a full live graph run |
| ⬜ | **LangSmith tracing-based evals** | LangSmith dep exists but no programmatic eval traces in the codebase |
| ⬜ | **Regression snapshot tests** | Eval JSONL is diffable but no automated regression assertion framework |

## 9 — Observability & Telemetry

| Status | Skill | Evidence / Notes |
|--------|-------|------------------|
| ✅ | **LangChain callback handler for telemetry** | `TelemetryCallbackHandler` logs token counts, model name, node name, errors per LLM call |
| ✅ | **Structured logging with `extra={}` fields** | Node tracer and telemetry emit machine-parseable structured fields |
| ✅ | **Per-node timing** | `node_trace` decorator records `elapsed_ms` per node execution |
| ✅ | **Dedicated log handler for node traces** | Separate FileHandler + StreamHandler with extended format for node tracer logger |
| ⬜ | **OpenTelemetry / distributed tracing** | No OTel spans; logging-only observability |
| ⬜ | **Dashboard / Grafana integration** | No metrics export |
| ⬜ | **LangSmith tracing in production** | LangSmith env vars referenced but not actively wired for production tracing |

## 10 — Storage & Retrieval

| Status | Skill | Evidence / Notes |
|--------|-------|------------------|
| ✅ | **Vector store with pgvector** | Fragments and claims stored with `vector(384)` embeddings; cosine similarity search |
| ✅ | **Local embeddings (fastembed/ONNX)** | `Embedder` wraps `sentence-transformers/all-MiniLM-L6-v2` — no external embedding API needed |
| ✅ | **Dual-write persistence (JSONL + Postgres)** | Every repository fans out to both `JsonlGateway` and `PgGateway` |
| ✅ | **Repository pattern** | `FragmentRepository`, `TranscriptRepository`, `ThreadsRepository`, `InsightsRepository`, `SubjectsRepository`, `CaptureRepository`, `UserProfileRepository` |
| ✅ | **RAG-style retrieval in graph** | `retrieve_history` node vector-searches fragments by latest human message + intent tags |
| ✅ | **Claim-based retrieval** | Subjects repo routes new fragments to candidate subjects by embedding similarity |
| ⬜ | **Hybrid search (keyword + vector)** | Vector-only; no BM25 or full-text search fusion |
| ⬜ | **Document loaders / external data ingestion** | All data comes from conversation; no file/URL/API ingestion |

## 11 — Architecture & Production Readiness

| Status | Skill | Evidence / Notes |
|--------|-------|------------------|
| ✅ | **FastAPI with lifespan dependency management** | `@asynccontextmanager async def lifespan(app)` builds full dep graph at startup |
| ✅ | **Terminal + HTTP dual interface** | Same graph backend serves `main.py` (CLI) and `api/main.py` (HTTP) |
| ✅ | **React/Vite frontend (SSE consumer)** | `journal_chat_app/` — TypeScript + React + Tailwind + Vite |
| ✅ | **Pydantic Settings with `.env` loading** | `Settings(BaseSettings)` with `SettingsConfigDict`, secret masking, LRU cache |
| ✅ | **Graph diagram generation** | `conversation.get_graph().draw_mermaid_png()` outputs `graph.png` at startup |
| ✅ | **Role-based model selection** | `LLM_ROLE_CONFIG` maps roles to model labels; easy to swap models per role |
| ⬜ | **Containerized deployment (Docker / K8s)** | No Dockerfile or Helm chart |
| ⬜ | **Authentication / multi-user** | Single-user design; no auth layer |
| ⬜ | **Rate limiting / guardrails** | No request-level rate limiting or content guardrails |
| ⬜ | **CI/CD pipeline** | No GitHub Actions / CI config |

## 12 — Advanced LangGraph Patterns (Not Yet Explored)

| Status | Skill | Description |
|--------|-------|-------------|
| ⬜ | **ReAct agent with tool use** | Agent reasons about which tool to call, observes the result, and iterates |
| ⬜ | **Tool nodes (`ToolNode`)** | Native LangGraph tool nodes that bind LangChain tools to graph execution |
| ⬜ | **Human-in-the-loop (`interrupt`)** | Graph pauses execution, waits for human approval, then resumes |
| ⬜ | **`Command` API for dynamic control flow** | LangGraph `Command` objects for goto / resume / update from within nodes |
| ⬜ | **Multi-agent collaboration** | Multiple specialized agents coordinating via shared state or message passing |
| ⬜ | **Hierarchical agent teams** | Supervisor agent dispatching to specialist sub-agents |
| ⬜ | **Plan-and-execute pattern** | Planner agent creates a task list; executor agent works through it step by step |
| ⬜ | **Reflection / self-critique loop** | Agent evaluates its own output and iterates until quality threshold is met |
| ⬜ | **Memory services (LangGraph Store / shared memory)** | Cross-session memory via LangGraph's built-in memory primitives |
| ⬜ | **Streaming tool calls / intermediate steps** | Streaming partial tool-call arguments as they arrive |
| ⬜ | **Cross-thread state sharing** | Sharing state or context between different conversation threads |
| ⬜ | **Dynamic graph construction at runtime** | Building graph topology based on runtime input |
| ⬜ | **LangGraph Cloud / Studio deployment** | Deploying to LangGraph's managed cloud platform |

---

## Summary

| Category | Done | Remaining | Coverage |
|----------|------|-----------|----------|
| Graph Fundamentals | 7 | 2 | 78% |
| State Design | 5 | 0 | 100% |
| Node Patterns | 6 | 0 | 100% |
| Routing & Control Flow | 5 | 1 | 83% |
| Checkpointing | 4 | 2 | 67% |
| LLM Integration | 7 | 5 | 58% |
| Streaming & Transport | 4 | 2 | 67% |
| Testing & Evaluation | 7 | 3 | 70% |
| Observability | 4 | 3 | 57% |
| Storage & Retrieval | 6 | 2 | 75% |
| Architecture | 6 | 4 | 60% |
| Advanced Patterns | 0 | 13 | 0% |
| **Total** | **61** | **37** | **62%** |

> **Bottom line:** You have strong, production-quality coverage of graph
> fundamentals, state design, node patterns, routing, persistence, and
> testing. The biggest growth areas are **tool use / ReAct**, **human-in-the-loop**,
> **multi-agent coordination**, and **deployment/ops maturity**.
