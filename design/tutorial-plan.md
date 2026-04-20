# Journal Agent — Tutorial Build Plan

## Context

Chris designed a journal-based LLM agent through a conversation with ChatGPT (captured in `design/*.md`). The vision: a persistent, context-aware conversation partner that remembers everything, extracts insights, and thinks with accumulated knowledge.

This project already has a working `conversation_engine` (LangGraph-based validation loop). Rather than extending that, we're building the journal agent standalone — Chris writes every line, with tutorial-level guidance at each step.

**Goal:** Build a demo-ready journaling agent that a CTO would consider evidence of senior-level AI engineering skill. Phases 1–9 are complete (correctness). The remaining phases migrate to production infrastructure, add derived insights, harden for resilience, and prepare the demo.

**Demo thesis:** The system should demonstrate three user-visible moments — "it remembers," "it notices patterns," "it adapts to me" — backed by architecture that bears professional scrutiny.

**Project location:** `journal_agent/` package alongside `conversation_engine/` in this repo.

---

## Phase 1: Hello LLM
**You'll learn:** OpenAI API basics, message format, environment variables

Build a single Python script that sends one message to the LLM and prints the response. ~15 lines of code.

- Create `journal_agent/` package with `__init__.py`
- Create `journal_agent/main.py`
- Use `langchain-openai` (already a dependency) to call ChatOpenAI
- Read API key from environment
- Send a hardcoded message, print the response

**Test it:** Run `python -m journal_agent.main` and see a response.

---

## Phase 2: Conversation Loop
**You'll learn:** While loops, message lists, the chat message format (system/human/ai), graceful exit

Turn the one-shot script into a loop: input → LLM → print → repeat.

- Add a `while True` loop with `input()` for user text
- Keep a `messages: list` that accumulates HumanMessage/AIMessage each turn
- Pass full message history to the LLM each call
- Add a quit command (`/quit` or `exit`)
- Add a system message that sets basic personality

**Test it:** Have a multi-turn conversation. Verify the LLM remembers what you said 3 turns ago.

---

## Phase 3: Persistence — Save and Load Sessions
**You'll learn:** JSON serialization, file I/O, Pydantic models, session identity

Save conversations to disk so you can pick up where you left off.

- Create `journal_agent/models.py` — define `Turn` (Pydantic model: role, content, timestamp)
- Create `journal_agent/storage.py` — `save_session(session_id, turns)` and `load_session(session_id)`
- Store as JSON files in a `data/sessions/` directory
- Generate a session_id (timestamp or UUID)
- On startup, option to resume a previous session or start new

**Test it:** Have a conversation, quit, restart, resume — verify the LLM has the prior context.

---

## Phase 4: Rebuild as LangGraph
**You'll learn:** StateGraph, TypedDict state, nodes as functions, edges, conditional routing, compile + invoke

Rebuild the exact same behavior using LangGraph. Same input/output, but now the framework manages state and flow.

- Create `journal_agent/graph/state.py` — `JournalState(TypedDict)` with messages, session_id, status
- Create `journal_agent/graph/nodes.py` — `get_input()`, `respond()`, `save_turn()`
- Create `journal_agent/graph/builder.py` — wire nodes: START → get_input → respond → save_turn → route
- Route: if user said quit → END, else → get_input
- Run with `graph.invoke(initial_state)`

**Concept focus:** Explain what LangGraph gives you over the raw while loop — state management, checkpointing potential, visual graph, composability. Show the graph visualization.

**Test it:** Same conversation works as before, but now through LangGraph.

---

## Phase 5: Fragment Extraction
**You'll learn:** Structured LLM output, Pydantic output parsing, the "write path" concept

After each turn, use the LLM to break the exchange into atomic fragments with tags.

- Add to `journal_agent/models.py` — `Fragment` (Pydantic: id, content, tags, summary, timestamp, source_turn_id)
- Create `journal_agent/extraction.py` — `extract_fragments(turn_text) -> list[Fragment]`
  - Uses a second LLM call with a structured output prompt
  - Prompt: "Break this into atomic ideas. For each, provide content, a 1-sentence summary, and 1-3 tags"
- Add `extract` node to the graph (runs after `save_turn`)
- Store fragments alongside sessions in `data/fragments/`

**Parallelization note:** the per-thread fan-out in the classifier/extractor is synchronous by design for this phase. Phase 11 converts it to `asyncio.gather`. Correctness first, then hardening.

**Concept focus:** This is the "write path" from the design — every conversation turn produces durable, searchable knowledge.

**Test it:** Have a conversation, then inspect the fragment files. Verify fragments are meaningful and well-tagged.

---

## Phase 6: Simple Retrieval
**You'll learn:** Embedding basics, cosine similarity, the "read path" concept, vector search

Before responding, search stored fragments for relevant context and include them in the prompt.

- Create `journal_agent/retrieval.py` — `search_fragments(query, top_k=3) -> list[Fragment]`
- Start simple: keyword/tag matching against stored fragments
- Then upgrade: generate embeddings (OpenAI embeddings API), store alongside fragments, use cosine similarity
- Add `retrieve` node to the graph (runs before `respond`)
- Pass retrieved fragments into the LLM prompt as "Relevant prior context: ..."

**Concept focus:** This is RAG at its simplest. You retrieve context to enrich the LLM's thinking. The design doc's principle: "RAG gives ingredients — Context Builder makes the meal."

**Test it:** Talk about a topic, quit, start new session, mention the topic again — verify the LLM references things from the previous session.

---

## Phase 7: Context Builder
**You'll learn:** Prompt engineering, token budgets, layered prompt assembly

Formalize how the LLM prompt is assembled. This is the heart of the system from the design doc.

- Create `journal_agent/context_builder.py` — `build_context(state) -> list[BaseMessage]`
- Assemble layers in order:
  1. System prompt (personality)
  2. Retrieved fragments (from Phase 6)
  3. Recent conversation turns (last N)
  4. Current user input
- Add token budget awareness: count tokens per section, trim if over limit
- Replace ad-hoc prompt building in `respond` node with `build_context()`

**Concept focus:** Context is *constructed*, not just retrieved. Same fragments, different assembly = different quality response.

**Test it:** Add logging that shows what context was assembled. Verify it's selecting relevant fragments and not dumping everything.

---

## Phase 8: Intent Detection
**You'll learn:** Classification with LLMs, enum types, conditional behavior

Classify what the user is doing each turn and use that to shape retrieval and response.

- Create `journal_agent/intent.py` — `detect_intent(user_input, recent_context) -> Intent`
- Intent is an enum: `design`, `exploration`, `reflection`, `evaluation`, `retrieval`
- Use a lightweight LLM call (or pattern matching for V1)
- Add `detect_intent` node to graph (runs early, before retrieve)
- Intent influences:
  - Whether RAG is triggered at all
  - How many fragments to retrieve
  - System prompt additions ("user is in design mode, be structured...")

**Parallelization note:** intent runs every turn, so latency matters. Keep V1 synchronous for correctness; Phase 11 addresses parallel fan-out (e.g., running intent + retrieval concurrently).

**Concept focus:** Intent drives everything downstream. Same memory, different intent → different context → different response.

**Test it:** Say "let's design a system" vs "what did we talk about last time?" vs "I'm just thinking out loud" — verify intent is classified correctly and response style changes.

### Design guidance

At its core this is a classifier node — but the *interesting* part is what you do with the label, not the classification itself. The classifier answers: *"what is the user trying to do this turn?"* → one of the enum values. The value comes from downstream branching that reads that label:

- **Router decision:** `exploration` → maybe skip RAG entirely; `retrieval` → run retrieval with a wider net
- **Retrieval params:** `design` wants 3 tightly-relevant fragments; `reflection` might want 8 loosely-related ones
- **Prompt shape:** ContextBuilder appends an intent-specific instruction block ("user is in design mode, be structured and concrete")

The classifier itself is boring — the architecture is where the learning lives: *how does one upstream signal reshape multiple downstream nodes?* That's why Phase 8 sits before parallelization (Phase 11): you need the signal to exist before you can fan out on it.

### The canonical pattern: "Routing"

This is a named pattern — **Routing**, from Anthropic's *Building Effective Agents* post. It's one of the five canonical agent patterns (prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer). Worth skimming that post before coding — it'll frame the whole curriculum, not just this phase.

### Practical patterns that save you from common mistakes

1. **Keep the taxonomy small (5–7 intents max).** Classifiers degrade fast past that. The current 5 is right-sized.

2. **Always include a fallback/unknown intent.** Force-choosing among 5 labels when the real answer is "none of these" produces confident-but-wrong labels. Better: 6 labels with `unknown` as the escape hatch.

3. **Separate classification from routing.** The classifier returns the label; a *router function* (or LangGraph conditional edge) consumes it. Strategy pattern — swap routing logic without retraining/retuning the classifier.

4. **Table-driven config, not if/else.** Map intent → `{top_k, max_distance, prompt_suffix}` in a dict. Declarative, easy to tune, easy to test:

   ```python
   INTENT_POLICY = {
       "design":      {"top_k": 3, "max_distance": 1.0, "prompt_suffix": "Be structured."},
       "exploration": {"top_k": 0, ...},  # skip retrieval
       ...
   }
   ```

5. **Log the classification (with confidence if you have it).** You'll want to review misclassifications later to refine the taxonomy — hard to do without a trace.

**What you don't need to learn ahead of time:** the classifier mechanics (prompt + structured output — you've done that). The downstream design is a judgment call — code the mechanics, make your own calls, iterate.

**NOTES:**
 ---                                                                                                                                                                                                                                               
  2. Chain-of-thought / self-critique scaffolds                                                                                                                                                                                                       
                                                                                                                                                                                                                                                    
  What it is
  Instead of asking the model to produce the final JSON in one leap, you force it to reason in steps or critique its own work before finalizing. The reasoning is still the LLM doing LLM things — but more tokens means more computation means more
  chance of getting it right.                                                                                                                                                                                                                         
   
  Why it works                                                                                                                                                                                                                                        
  LLMs "think" by generating tokens. A long reasoning trace gives the model room to work through sub-problems. It's also how you actually did this task — you didn't one-shot it; you read, identified threads, evaluated each against the taxonomy,
  then wrote. Making that sequence explicit lets a smaller model follow the same path.                                                                                                                                                                
   
  How you'd actually do it for the classifier                                                                                                                                                                                                         
                                                                                                                                                                                                                                                    
  The one-shot approach we've been using:                                                                                                                                                                                                             
  transcript → [prompt with all rules] → ClassifiedExchangeList                                                                                                                                                                                     

  The structured-CoT approach:                                                                                                                                                                                                                        
  transcript → [prompt: "list the topical threads with names and exchange_ids"] → thread_list
  thread_list + transcript → [prompt: "for each thread, walk the taxonomy and list applicable tags"] → tag_list                                                                                                                                       
  thread_list + tag_list + transcript → [prompt: "for each thread, write voice-preserving summaries"] → summaries                                                                                                                                     
  all of above → [assemble final JSON]                                                                                                                                                                                                                
                                                                                                                                                                                                                                                      
  Each step is simpler than the whole. A small model that couldn't do the full task can often do each sub-step. Since you're using LangGraph, each step becomes a node — which is actually a clean architectural fit.                                 
                                                                                                                                                                                                                                                      
  Self-critique variant (cheaper first attempt)                                                                                                                                                                                                       
  - Same prompt as now, model produces initial ClassifiedExchangeList.                                                                                                                                                                                
  - Second call: "Here's the transcript and your classification. Review it. For each record, check: did you miss any applicable taxonomy tags? Did you preserve specific phrases? Did you miss any topical threads?" → revised output.                
  - Costs 2× inference, often closes a real quality gap.                                                                                                                                                                                            
                                                                                                                                                                                                                                                      
  Trade-offs                                                                                                                                                                                                                                        
  - More tokens → more cost per classification (2–5× on small models).                                                                                                                                                                                
  - More latency.                                                                                                                                                                                                                                   
  - More moving parts to debug — failures at step 1 cascade.                                                                                                                                                                                          
  - But: no training infrastructure, works on models you already run locally, easy to iterate on.                                                                                                                                                     
                                                                                                                                                                                                                                                      
  When it makes sense for you                                                                                                                                                                                                                         
  If you ever want to move the classifier off GPT-4o without fine-tuning. Structured CoT on a local 7–8B model + LangGraph-based decomposition can realistically hit 80–90% of one-shot-GPT-4o quality. The architecture you're already building      
  naturally supports this — adding an "identify_threads" node upstream of "classify_threads" is a one-file change.                                                                                                                                    
                                                                                                                                                                                                                                                      

---

## Phase 9: Personality and User Profile
**You'll learn:** System prompt design, lightweight user modeling, preference tracking

Shape the agent's personality and adapt to the user over time.

- Create `journal_agent/profile.py` — `UserProfile` (Pydantic: preferences, themes, traits)
- Store profile in `data/profile.json`
- Update profile periodically (every N turns, use LLM to summarize emerging patterns)
- Enhance system prompt with profile data: "This user prefers structured responses and is interested in X, Y, Z"
- Design the "Drew-like" personality prompt

**Concept focus:** The profile tunes behavior. The personality prompt creates consistency. Together they make the agent feel like it "knows" you.

**Test it:** Have several conversations. Verify the profile builds up. Verify responses adapt to stated preferences.

---

## Phase 10: Infrastructure Migration
**You'll learn:** Docker Compose, Postgres + pgvector, database schema design, replacing file-based storage with SQL, user scoping

Migrate from file-based storage (JSONL + Chroma) to a production-grade stack. This unblocks cross-session queries, multi-user support, and consolidated vector search — all of which are prerequisites for the insight pipeline and the demo.

**Why now:** The insight pipeline (Phase 12) needs cross-session aggregation queries that are painful against JSONL files (glob + parse + filter). Postgres with pgvector consolidates structured queries and vector search into one system, eliminating the impedance mismatch between Chroma and the relational store.

### Stack

```yaml
# docker-compose.yml
services:
  postgres:
    image: pgvector/pgvector:pg16
    volumes:
      - pgdata:/var/lib/postgresql/data

  neo4j:                              # provisioned for future graph-of-ideas
    image: neo4j:5-community          # not wired yet — shows architectural forethought
    volumes:
      - neo4jdata:/data

  app:
    build: .
    depends_on: [postgres, neo4j]
    environment:
      - DATABASE_URL=postgresql://...
      - NEO4J_URI=bolt://neo4j:7687
      - OPENAI_API_KEY=${OPENAI_API_KEY}

volumes:
  pgdata:
  neo4jdata:
```

### Schema

```sql
CREATE TABLE users (
    user_id     TEXT PRIMARY KEY,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE sessions (
    session_id  TEXT PRIMARY KEY,
    user_id     TEXT REFERENCES users(user_id),
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE fragments (
    fragment_id TEXT PRIMARY KEY,
    session_id  TEXT REFERENCES sessions(session_id),
    user_id     TEXT REFERENCES users(user_id),
    content     TEXT NOT NULL,
    tags        JSONB DEFAULT '[]',
    embedding   vector(1536),
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_fragments_user    ON fragments(user_id);
CREATE INDEX idx_fragments_tags    ON fragments USING GIN(tags);
CREATE INDEX idx_fragments_time    ON fragments(created_at);
CREATE INDEX idx_fragments_embed   ON fragments USING hnsw(embedding vector_cosine_ops);

CREATE TABLE insights (
    insight_id   TEXT PRIMARY KEY,
    user_id      TEXT REFERENCES users(user_id),
    tag          TEXT NOT NULL,
    fragment_ids JSONB DEFAULT '[]',
    content      TEXT NOT NULL,
    confidence   FLOAT CHECK (confidence BETWEEN 0 AND 1),
    supersedes   TEXT REFERENCES insights(insight_id),
    created_at   TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_insights_user     ON insights(user_id);
CREATE INDEX idx_insights_current  ON insights(user_id, tag) WHERE supersedes IS NULL;

CREATE TABLE user_profiles (
    user_id      TEXT PRIMARY KEY REFERENCES users(user_id),
    profile_data JSONB NOT NULL DEFAULT '{}',
    updated_at   TIMESTAMPTZ DEFAULT now()
);
```

### Tasks
- Docker Compose with Postgres (pgvector), Neo4j (provisioned only), app service
- Postgres schema: `users`, `sessions`, `fragments` (with `vector(1536)` column), `insights`, `user_profiles`
- `PostgresStore` replacing `JsonStore` — same public interface, SQL backend
- `PgVectorStore` replacing `VectorStore` (Chroma) — `add_fragments()` and `search_fragments()` with filtered vector search
- `user_id` scoping — flow `user_id` through the graph alongside `session_id`; all queries filter on it
- Data migration script: dump existing JSONL → INSERT into Postgres

**What gets dropped:** Chroma, JSONL file storage. Your event log (raw session JSONL) can optionally stay as an immutable backup — Postgres becomes the primary store.

**Test it:** Run existing test suite against the new stores. Verify `docker compose up` brings up the full stack. Verify two different user_ids produce isolated data.

---

## Phase 11: Hardening
**You'll learn:** Retry with backoff, per-call error isolation, cost tracking, `asyncio.gather` for LLM fan-out

Harden the pipeline *before* adding the insight feature. Get retry and error handling in place so that transient LLM failures don't derail development or the demo.

### Tasks

**Critical (do before Phase 12):**

- **`tenacity` retry on LLM calls** — decorator on `LLMClient.invoke` / `structured`. Exponential backoff for 429s, 5xx, timeouts. ~5 lines of decorator code but prevents demo embarrassment.
- **Per-call error isolation** — one bad LLM response (rate limit, malformed output) should log and skip, not crash the pipeline. The end-of-session fan-out (classifier → extractor → fragments) must survive partial failures.

**Important (do during or after Phase 12):**

- **Cost tracking** — sum input/output tokens per node per session, write to a summary table or file. During the CTO walkthrough: "This session cost $0.03 across 4 LLM calls." Shows production cost awareness.
- **`asyncio.gather` on end-of-session pipeline** — convert per-thread loops (classifier, extractor, insight pipeline) to concurrent fan-out. Measure latency before/after. Not critical for the demo (runs after `/quit`) but impressive in the architecture walkthrough.

**Test it:** Induce a failure (mock a 429 from OpenAI). Verify: the failed call retries, and if it still fails, the rest of the batch continues cleanly. Verify cost log is populated after a session.

---

## Phase 12: Derived Insights (Tag-Cluster)
**You'll learn:** Batch pipeline design, strategy pattern, incremental processing, materialized views, feeding derived data back into the conversation

Build the "derived memory" layer — the system learns patterns across all conversations. This is the feature that makes the demo go from "good memory" to "it actually understands me."

**Architectural framing:** The insight layer is a *materialized view* over the fragment store — batch-derived, incrementally refreshed, served into the context assembly path at query time.

### 1. New schemas

```python
class Insight(BaseModel):
    """A derived observation synthesised from a cluster of Fragments."""
    insight_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tag: str                          # the cluster's primary tag
    fragment_ids: list[str]           # provenance: which fragments contributed
    content: str                      # the derived pattern / tension / theme
    confidence: float = Field(ge=0, le=1)  # LLM self-rated
    created_at: datetime = Field(default_factory=datetime.now)
    supersedes: str | None = None     # insight_id this replaces on re-run

class InsightList(BaseModel):
    """Wrapper for structured LLM output."""
    insights: list[Insight]
```

`supersedes` enables **incremental updates**: when a cluster grows and you
re-derive, the new insight links to the one it replaces. Old insights stay
in the store for auditability; the read path filters to the latest via
`WHERE supersedes IS NULL`.

### 2. Aggregation strategy protocol

```python
class AggregationStrategy(Protocol):
    """Seam between the pipeline and the partitioning logic.

    Implement for each insight variant:
      - TagClusterStrategy   (this phase — partition by primary tag)
      - SessionWindowStrategy (future — partition by session, then diff)
    """
    def partition(self, fragments: list[Fragment]) -> dict[str, list[Fragment]]: ...
    def filter(self, groups: dict[str, list[Fragment]]) -> dict[str, list[Fragment]]: ...
    def build_prompt(self, key: str, fragments: list[Fragment]) -> str: ...
```

### 3. Tag-cluster strategy

```python
class TagClusterStrategy:
    min_cluster_size: int = 3

    def partition(self, fragments: list[Fragment]) -> dict[str, list[Fragment]]:
        clusters: dict[str, list[Fragment]] = {}
        for f in fragments:
            key = f.tags[0].tag if f.tags else "untagged"
            clusters.setdefault(key, []).append(f)
        return clusters

    def filter(self, groups: dict[str, list[Fragment]]) -> dict[str, list[Fragment]]:
        return {k: v for k, v in groups.items() if len(v) >= self.min_cluster_size}

    def build_prompt(self, key: str, fragments: list[Fragment]) -> str:
        fragment_block = "\n".join(
            f"- [{f.fragment_id}] {f.content}" for f in fragments
        )
        return (
            f"You are analyzing {len(fragments)} journal fragments about '{key}'.\n\n"
            f"Fragments:\n{fragment_block}\n\n"
            "Identify recurring patterns, tensions, or evolving themes across "
            "these fragments.  For each insight:\n"
            "- State the pattern in one clear sentence.\n"
            "- Rate your confidence (0.0–1.0) that this is a real pattern, not noise.\n"
            "- List the fragment_ids that support it.\n\n"
            "Return your answer as a JSON object matching the InsightList schema."
        )
```

### 4. Insight pipeline

```python
class InsightPipeline:
    """Stateless batch pipeline: load → partition → derive → store."""

    def __init__(self, llm: LLMClient, store: InsightStore):
        self.llm = llm
        self.store = store

    def run(
        self,
        fragments: list[Fragment],
        strategy: AggregationStrategy,
        watermark: datetime | None = None,
    ) -> list[Insight]:
        if watermark:
            fragments = [f for f in fragments if f.timestamp > watermark]

        groups = strategy.partition(fragments)
        groups = strategy.filter(groups)

        all_insights: list[Insight] = []
        for key, cluster in groups.items():
            prompt = strategy.build_prompt(key, cluster)
            structured_llm = self.llm.structured(InsightList)
            result: InsightList = structured_llm.invoke([
                SystemMessage(content=prompt),
            ])
            for insight in result.insights:
                insight.tag = key
                insight.fragment_ids = [f.fragment_id for f in cluster]
            all_insights.extend(result.insights)

        self.store.save(all_insights)
        return all_insights
```

### 5. Storage

Insight storage is backed by the `insights` Postgres table (schema in Phase 10).
`InsightStore` wraps it with the same interface pattern as other stores:

```python
class InsightStore:
    def save(self, insights: list[Insight]) -> None: ...
    def load_all(self, user_id: str) -> list[Insight]: ...
    def load_current(self, user_id: str) -> list[Insight]:
        """Latest version of each insight (WHERE supersedes IS NULL)."""
        ...
```

### 6. Graph integration

The insight pipeline runs in the **end-of-session pipeline**, after fragments
are saved:

```
End-of-session pipeline (updated):
    save_transcript → exchange_decomposer → save_threads
      → thread_classifier → save_classified_threads
      → thread_fragment_extractor → save_fragments
      → generate_insights → END
         ^^^^^^^^^^^^^^^^^^
         NEW NODE
```

**Node prototype:**

```python
def make_generate_insights(
    llm: LLMClient,
    insight_store: InsightStore | None = None,
) -> Callable[..., dict]:
    insight_store = insight_store or InsightStore()
    pipeline = InsightPipeline(llm=llm, store=insight_store)
    strategy = TagClusterStrategy()

    @node_trace("generate_insights")
    def generate_insights(state: JournalState) -> dict:
        try:
            all_fragments = fragment_store.load_all(user_id=state["user_id"])
            existing = insight_store.load_current(user_id=state["user_id"])
            watermark = max((i.created_at for i in existing), default=None)
            insights = pipeline.run(all_fragments, strategy, watermark=watermark)
            return {"status": Status.INSIGHTS_GENERATED}
        except Exception as e:
            logger.exception("Insight generation failed")
            return {"status": Status.ERROR, "error_message": str(e)}

    return generate_insights
```

### 7. Feeding insights into the conversation

Insights are retrieved alongside fragments during the conversation loop.
The `retrieve_history` node gains an insight lookup:

```
retrieve_history node (updated):
    1. Vector search for relevant fragments     (existing)
    2. Load current insights matching top tags   (NEW)
    3. Return both in state
```

Context Builder assembles them as a separate layer in the system message:

```
SystemMessage content:
    {prompt}
    <retrieved_context>...</retrieved_context>    ← fragments (existing)
    <insights>...</insights>                      ← derived insights (NEW)
```

### 8. Extensibility — future strategies

The `AggregationStrategy` protocol is the seam for future insight types:

| Future strategy | What it adds | Effort to plug in |
|---|---|---|
| **SessionWindowStrategy** (temporal drift) | Partitions by session, generates digests, diffs consecutive pairs for emerging/persistent/fading patterns | New strategy class + `SessionDigest` model + `DiffPass`. Zero changes to pipeline or store. |
| **GraphClusterStrategy** (idea connections) | Uses embedding similarity as pre-filter, LLM extracts edges, connected components become insight clusters | New strategy class + `IdeaEdge` model. Wires into Neo4j. |

Both plug into the existing `InsightPipeline` with no structural changes.

**Test it:** After several sessions, run insights. Verify they capture real patterns, not noise. Verify `supersedes` chains work (re-run after adding more sessions — old insights get replaced, not duplicated).

---

## Phase 13: Demo Preparation
**You'll learn:** Seed data generation, demo reliability, architecture storytelling, presentation

Ship the demo. This phase is about making the product reliable and presentable, not adding features.

### Tasks

- **Seed data script** — generates 8–10 realistic multi-topic sessions so the demo is warm from the first interaction. Without this, a demo user must invest 30+ minutes before insights have anything to work with.
- **Prompt polish** — tune the personality prompt, insight surfacing language, and profile adaptation. Iterate with real conversations over 2–3 rounds.
- **Architecture doc** — one-page diagram + 10 bullet points explaining design decisions. This is what the CTO takes away after the meeting. Include: event-sourced write path, materialized view insight layer, strategy pattern extensibility, pgvector consolidation, Context Builder separation of concerns.
- **Git history cleanup** — squash experiments. The commit log should read like a professional progression.
- **Rehearse** — the demo itself takes 5 minutes (three moments: remembers, notices, adapts). The architecture walkthrough takes 15. The questions take 30. Practice all three.

### The three demo moments

| Moment | What happens | What the CTO sees |
|---|---|---|
| **"It remembers"** | User mentions something from a previous session. Agent references it naturally. | Cross-session RAG with pgvector, semantic retrieval as a passive feed |
| **"It notices"** | Agent volunteers a pattern: "I notice you keep coming back to the tension between X and Y" | Materialized insight pipeline, strategy pattern, incremental processing |
| **"It adapts"** | User says "be more direct." Next response shifts tone. | Profile scanner node, parametric prompts, separation of concerns |

### CTO walkthrough points

1. Event-sourced architecture: immutable write path → derived stores → read path
2. Strategy pattern in the insight pipeline: `AggregationStrategy` protocol, extensible without refactoring
3. Context Builder separation of concerns: nodes own their prompts, builder does pure assembly
4. Test suite: 196+ tests, edge cases, behavior-driven
5. Error handling: `@node_trace`, `Status.ERROR` propagation, tenacity retry
6. Cost tracking: per-node token accounting
7. Infrastructure: `docker compose up`, Postgres + pgvector, Neo4j provisioned

---

## Graph Topology Evolution

```
Phase 2:   input → respond → loop
Phase 4:   START → get_input → respond → save_turn → route → END
Phase 5:   START → get_input → respond → save_turn → extract → route → END
Phase 6:   START → get_input → retrieve → respond → save_turn → extract → route → END
Phase 8:   START → get_input → detect_intent → retrieve → respond → save_turn → extract → route → END
Phase 9:   START → get_input → intent_classifier → profile_scanner → retrieve → respond → ... → END
Phase 12:  (end-of-session) ... → save_fragments → generate_insights → END
```

## Project Structure (final)

```
journal_agent/
├── __init__.py
├── main.py                      # Entry point
├── model/
│   └── session.py               # Turn, Fragment, Insight, UserProfile, ScoreCard, etc.
├── configure/
│   ├── config_builder.py        # Environment + defaults
│   ├── context_builder.py       # Layered prompt assembly + token budgeting
│   └── prompts/                 # Parametric prompt templates per node
│       ├── __init__.py          # get_prompt(key, state) registry
│       ├── conversation.py
│       ├── intent_classifier.py
│       ├── profile_scanner.py
│       └── socratic.py
├── graph/
│   ├── state.py                 # JournalState (TypedDict)
│   ├── graph.py                 # Wire the graph, make_get_ai_response
│   └── nodes/
│       ├── classifier.py        # intent_classifier, profile_scanner
│       ├── insights.py          # generate_insights (end-of-session)
│       └── ...                  # save nodes, retrieval, extraction
├── insights/
│   ├── strategy.py              # AggregationStrategy protocol
│   ├── tag_cluster.py           # TagClusterStrategy
│   └── pipeline.py              # InsightPipeline
├── storage/
│   ├── postgres_store.py        # PostgresStore (replaces JsonStore)
│   ├── pgvector_store.py        # PgVectorStore (replaces VectorStore/Chroma)
│   ├── insight_store.py         # InsightStore
│   ├── profile_store.py         # UserProfileStore
│   └── exchange_store.py        # TranscriptStore
├── tests/
│   ├── test_models.py
│   ├── test_context_builder.py
│   ├── test_nodes.py
│   └── ...
├── docker-compose.yml
└── seed_data.py                 # Generate realistic demo sessions
```

## Verification

After each phase:
1. Run the agent interactively and verify the new behavior
2. Run the full test suite: `python -m pytest journal_agent/tests/ -q --tb=short`
3. Review the code together — explain back what it does

End-to-end test after Phase 12 (correctness):
- Have 3+ conversations across sessions on different topics
- Verify: fragments are extracted, retrieval finds cross-session context, intent shapes responses, profile adapts, insights emerge
- Verify multi-user isolation: two user_ids produce completely independent data

End-to-end test after Phase 13 (demo readiness):
- Run seed data script, start fresh demo session as a new user
- Verify: insights are warm from session one, the three demo moments land
- Induce a failure (mock a 429). Verify: retry kicks in, pipeline survives partial failures

## Explicitly out of scope

- **Phase 10B** (temporal drift / session digests) — `AggregationStrategy` protocol supports it; build when real usage data exists
- **Phase 10C** (graph-of-ideas) — Neo4j is provisioned in Compose; build when edge extraction is worth the quadratic cost
- **Web UI beyond minimal** — a CTO doesn't evaluate CSS
- **Auth / multi-tenant** — user_id scoping is sufficient; auth is an infrastructure concern (Auth0/Clerk), not hand-rolled
- **Kafka / event streaming** — append-only Postgres tables serve the event-log role at this scale
