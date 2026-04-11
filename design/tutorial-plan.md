# Journal Agent — Tutorial Build Plan

## Context

Chris designed a journal-based LLM agent through a conversation with ChatGPT (captured in `design/*.md`). The vision: a persistent, context-aware conversation partner that remembers everything, extracts insights, and thinks with accumulated knowledge.

This project already has a working `conversation_engine` (LangGraph-based validation loop). Rather than extending that, we're building the journal agent standalone — Chris writes every line, with tutorial-level guidance at each step.

**Goal:** Build the journal agent in 10 phases. Each phase produces something runnable. Each phase teaches specific Python/LangGraph concepts.

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

**Concept focus:** Intent drives everything downstream. Same memory, different intent → different context → different response.

**Test it:** Say "let's design a system" vs "what did we talk about last time?" vs "I'm just thinking out loud" — verify intent is classified correctly and response style changes.

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

## Phase 10: Derived Insights
**You'll learn:** Async/periodic processing, aggregation, cross-session analysis

Build the "derived memory" layer — the system learns patterns across all conversations.

- Create `journal_agent/insights.py` — `generate_insights(fragments) -> list[Insight]`
- Run periodically (end of session or on-demand): analyze all fragments for themes, connections, trends
- Store insights alongside fragments
- Feed relevant insights into Context Builder (Phase 7) as an additional layer
- Add a `/insights` command that shows what the system has learned

**Concept focus:** This is where the system goes beyond memory into understanding. It's the difference between "I remember you said X" and "I notice you keep coming back to the tension between X and Y."

**Test it:** After several sessions, run insights. Verify they capture real patterns, not noise.

---

## Graph Topology Evolution

```
Phase 2:  input → respond → loop
Phase 4:  START → get_input → respond → save_turn → route → END
Phase 5:  START → get_input → respond → save_turn → extract → route → END
Phase 6:  START → get_input → retrieve → respond → save_turn → extract → route → END
Phase 8:  START → get_input → detect_intent → retrieve → respond → save_turn → extract → route → END
```

## Project Structure (final)

```
journal_agent/
├── __init__.py
├── main.py                  # Entry point
├── models.py                # Turn, Fragment, Intent, UserProfile, Insight
├── graph/
│   ├── __init__.py
│   ├── state.py             # JournalState (TypedDict)
│   ├── nodes.py             # get_input, detect_intent, retrieve, respond, save_turn, extract
│   └── builder.py           # Wire the graph
├── extraction.py            # Fragment extraction from turns
├── retrieval.py             # Search stored fragments (embeddings + tags)
├── context_builder.py       # Layered prompt assembly
├── intent.py                # Intent classification
├── profile.py               # User profile management
├── insights.py              # Derived insights generation
└── storage.py               # Session + fragment persistence
```

## Verification

After each phase:
1. Run the agent interactively and verify the new behavior
2. Write at least one small test for the new code
3. Review the code together — explain back what it does (solidifies learning)

End-to-end test after Phase 10:
- Have 3+ conversations across sessions on different topics
- Verify: fragments are extracted, retrieval finds cross-session context, intent shapes responses, profile adapts, insights emerge
