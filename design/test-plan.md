# Test Plan — Journal Agent

Partitioned by layer so each block can be tackled independently.
Layers 1–4 are pure logic (no mocking). Layers 5–7 require mocks/stubs.

**Recommended model allocation:**
- Layers 1–4 → Sonnet / GPT-4o (straightforward, stretches credits)
- Layers 5–7 → Opus (architectural judgment, full-graph context)

---

## Layer 1: Pydantic Models — `model/session.py`

**File:** `tests/test_models.py`
**Depends on:** nothing

| # | Test | What it proves |
|---|------|----------------|
| 1 | `UserProfile()` with no args produces valid instance | All defaults wired correctly |
| 2 | Two `UserProfile()` instances have independent `interests` / `pet_peeves` lists | Mutable-default safety (`default_factory`) |
| 3 | `UserProfile` round-trips through `model_dump_json()` → `model_validate_json()` | Serialization fidelity |
| 4 | `ContextSpecification()` defaults match config constants | Defaults sourced from config_builder |
| 5 | `Fragment`, `Exchange`, `ClassifiedExchange` auto-generate UUIDs | `default_factory` for IDs |
| 6 | `ScoreCard` accepts boundary floats (0.0, 1.0) | Validation range |
| 7 | `PromptKey` members match prompt file names | Enum ↔ file contract |
| 8 | `Turn.timestamp` defaults to now | Auto-timestamp |

---

## Layer 2: Config — `configure/settings.py`, `configure/config_builder.py`

**File:** `tests/test_config.py`
**Depends on:** nothing (mock env vars)

| # | Test | What it proves |
|---|------|----------------|
| 1 | `get_settings()` returns `Settings` instance with defaults | Baseline loading |
| 2 | `get_settings()` is cached (same object on second call) | `lru_cache` works |
| 3 | `get_settings.cache_clear()` → fresh instance | Test isolation path |
| 4 | `Settings.selected_model` injects API key from settings attrs | Key resolution logic |
| 5 | `Settings.selected_model` returns `None` when `llm_model` is `None` | Guard clause |
| 6 | `models` dict has an entry for every `LLMLabel` member | No orphan labels |
| 7 | `LLM_ROLE_CONFIG` keys are all strings, values are all valid `LLMLabel` | Type contract |
| 8 | `configure_environment()` returns `Settings` with `llm_model` populated | Smoke test (needs env mock) |
| 9 | All `DEFAULT_*` constants in `config_builder` are non-None (except `AI_NAME`) | No accidental nulls |

---

## Layer 3: Context Building — `configure/context_builder.py`

**File:** `tests/test_context_builder.py` (extend existing)
**Depends on:** Layer 1 models, prompts

*You already have good coverage here.* Additions:

| # | Test | What it proves |
|---|------|----------------|
| 1 | System message includes `UserProfile` data when profile is injected | Phase 9 integration point |
| 2 | Multiple prompt keys (`socratic`, `guidance`, `conversation`) each produce valid system messages | Prompt key → template wiring |
| 3 | Empty `retrieved_history` omits `<retrieved_context>` block | Clean output |

---

## Layer 4: Score Card — `configure/score_card.py`

**File:** `tests/test_score_card.py`
**Depends on:** Layer 1 models

| # | Test | What it proves |
|---|------|----------------|
| 1 | Each `Intent` enum value maps to a known `ContextSpecification` (or default) | No gaps in intent coverage |
| 2 | All 8 boolean triples (2³) produce a valid `Intent` | Exhaustive intent space |
| 3 | `resolve_scorecard_to_specification` with all scores at 0.0 → `OBSERVING` → socratic spec | Low-score path |
| 4 | `resolve_scorecard_to_specification` with all scores at 1.0 → `SEEKING_HELP` → default spec | High-score path |
| 5 | `resolve_scorecard_to_specification` propagates domain tags to spec | Domain passthrough |
| 6 | Threshold boundary: score exactly at threshold value → correct side | Edge case |

---

## Layer 5: Comms — `comms/llm_client.py`, `comms/llm_registry.py`

**File:** `tests/test_comms.py`
**Depends on:** Layer 2 config; needs mock LangChain chat models

| # | Test | What it proves |
|---|------|----------------|
| 1 | `LLMClient.chat()` forwards to underlying client's `.invoke()` | Delegation contract |
| 2 | `LLMClient.structured()` calls `.with_structured_output()` | Schema passthrough |
| 3 | `create_llm_client` raises `ValueError` for unknown provider | Guard clause |
| 4 | `build_llm_registry` creates clients for all roles in `LLM_ROLE_CONFIG` | Wiring completeness |
| 5 | `LLMRegistry.get("unknown_role")` falls back to `"conversation"` | Fallback behavior |
| 6 | `LLMRegistry.get()` raises `KeyError` when no conversation fallback | Error path |

**Mocking strategy:** Patch `ChatOpenAI`, `ChatAnthropic`, `ChatOllama` constructors to return a `MagicMock` with `.invoke()` and `.with_structured_output()`.

---

## Layer 6: Graph Nodes — `graph/nodes/classifier.py`, `graph/nodes/save_data.py`

**File:** `tests/test_nodes.py`
**Depends on:** Layer 1 models, Layer 5 comms (stub LLMClient)

### Classifier nodes (`classifier.py`)

| # | Test | What it proves |
|---|------|----------------|
| 1 | `make_exchange_decomposer` returns threads from stubbed LLM | Structured output → state mapping |
| 2 | `make_thread_classifier` returns classified threads | Tag propagation |
| 3 | `make_thread_fragment_extractor` returns fragments | Fragment construction from drafts |
| 4 | `make_intent_classifier` returns `ContextSpecification` via score card | End-to-end intent → spec |
| 5 | Each classifier node sets error status on LLM exception | Error handling |

### Save nodes (`save_data.py`)

| # | Test | What it proves |
|---|------|----------------|
| 1 | `make_save_transcript` writes to `JsonStore("transcripts")` | Store wiring |
| 2 | `make_save_threads` writes to `JsonStore("threads")` | Store wiring |
| 3 | `make_save_fragments_to_json` writes to `JsonStore("fragments")` | Store wiring |
| 4 | `make_save_fragments_to_vectordb` calls `vector_store.add_to_chroma_from_fragments` | Vector store delegation |
| 5 | Each save node returns `Status.ERROR` on exception | Error path |

**Mocking strategy:** For classifiers, create a stub `LLMClient` whose `.structured()` returns a mock runnable that produces a fixed Pydantic instance. For save nodes, use `tmp_path` fixture with patched `_resolve_project_root`.

---

## Layer 7: Storage — `storage/storage.py`, `storage/exchange_store.py`, `storage/vector_store.py`

**File:** `tests/test_storage.py`
**Depends on:** Layer 1 models; uses `tmp_path`

| # | Test | What it proves |
|---|------|----------------|
| 1 | `JsonStore.save_session` + `load_session` round-trips exchanges | File I/O fidelity |
| 2 | `JsonStore.save_session` appends to existing file | Append mode |
| 3 | `JsonStore.load_session` returns `None` for missing session | Missing file path |
| 4 | `JsonStore.get_last_session_id` returns newest file stem | Recency logic |
| 5 | `TranscriptStore.on_human_turn` / `on_ai_turn` builds exchanges | Turn accumulation |
| 6 | `VectorStore.fragment_to_chroma` ↔ `fragment_from_chroma` round-trip | Chroma serialization |
| 7 | `VectorStore.search_fragments` with `min_relevance` filters low matches | Relevance gating |

**Mocking strategy:** `tmp_path` + `monkeypatch` on `_resolve_project_root` for `JsonStore`. For `VectorStore`, use Chroma's ephemeral client (`chromadb.EphemeralClient()`).

---

## Layer 8 (optional): Integration — full graph

**File:** `tests/test_graph_integration.py`
**Depends on:** everything

| # | Test | What it proves |
|---|------|----------------|
| 1 | `build_journal_graph` returns a compiled graph | Graph construction |
| 2 | Single-turn invoke with stub LLM reaches `COMPLETED` status | Happy path |
| 3 | End-of-session pipeline (decompose → classify → extract → save) runs with stubs | Pipeline wiring |

**Mocking strategy:** Full stub registry (all roles return canned responses). Mock stores. This is Opus territory.

---

## Execution Order

```
Layer 1 → Layer 2 → Layer 4 → Layer 3 → Layer 7 → Layer 5 → Layer 6 → Layer 8
```

Layer 4 before 3 because score_card is simpler and context_builder already has tests.
Layer 7 before 5–6 because node tests need to mock storage, so verify storage works first.
