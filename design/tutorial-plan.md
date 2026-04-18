# Journal Agent — Tutorial Build Plan

## Context

Chris designed a journal-based LLM agent through a conversation with ChatGPT (captured in `design/*.md`). The vision: a persistent, context-aware conversation partner that remembers everything, extracts insights, and thinks with accumulated knowledge.

This project already has a working `conversation_engine` (LangGraph-based validation loop). Rather than extending that, we're building the journal agent standalone — Chris writes every line, with tutorial-level guidance at each step.

**Goal:** Build the journal agent in 11 phases. Each phase produces something runnable. Each phase teaches specific Python/LangGraph concepts. Phases 1–10 focus on correctness; Phase 11 hardens the whole pipeline for performance and resilience.

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

## Phase 10: Derived Insights
**You'll learn:** Async/periodic processing, aggregation, cross-session analysis

Build the "derived memory" layer — the system learns patterns across all conversations.

- Create `journal_agent/insights.py` — `generate_insights(fragments) -> list[Insight]`
- Run periodically (end of session or on-demand): analyze all fragments for themes, connections, trends
- Store insights alongside fragments
- Feed relevant insights into Context Builder (Phase 7) as an additional layer
- Add a `/insights` command that shows what the system has learned

**Parallelization note:** insight generation naturally fans out across fragments or themes. Keep V1 synchronous; Phase 11 parallelizes this along with the other fan-out points in the pipeline.

**Concept focus:** This is where the system goes beyond memory into understanding. It's the difference between "I remember you said X" and "I notice you keep coming back to the tension between X and Y."

**Test it:** After several sessions, run insights. Verify they capture real patterns, not noise.

---

## Phase 11: Performance & Resilience
**You'll learn:** `asyncio.gather` for LLM fan-out, graceful per-call failure handling, retry with backoff, rate-limit awareness, LangSmith observability review, light cost tracking

By Phase 10 you have a full pipeline with several places where LLM calls fan out: per-thread classification, per-thread extraction, batch embeddings, per-session insights. Each has been synchronous until now — correct by design, because correctness should precede scale. This phase converts correct code into code you could run at scale.

- Convert per-thread loops (classifier, extractor, insights) to `asyncio.gather` fan-out. Use your LLMClient's async variants (`ainvoke`). Measure latency before/after — this is where the win becomes tangible.
- Add per-call error handling: one poisoned LLM response (rate limit, malformed output, network blip) should log and skip, not kill the whole batch. This is the real-world version of the "one bad thread shouldn't take down the others" principle.
- Add retry with exponential backoff for transient failures (429s, 5xx, timeouts). Use a library (`tenacity`) rather than hand-rolling.
- Review LangSmith traces: what per-call information is useful, what's missing. Add timing and metadata where traces are thin.
- Optional: light cost tracking — sum tokens per node per session, write to a summary file so you can see where your money goes.

**Concept focus:** Hardening is a distinct activity from building. Your correctness-first approach (synchronous, loud failures) was right for learning; this phase converts correct code into scalable code. The pattern — ship correctness, then harden for scale — is how real systems get built professionally. Trying to do both at once is how projects get stuck.

**Test it:** Re-run sessions that work from earlier phases; verify parity (same outputs, faster). Induce a failure (e.g., feed a malformed exchange) and verify the rest of the batch continues cleanly. Compare end-to-end session latency with and without `asyncio.gather` on the classifier.

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

End-to-end test after Phase 10 (correctness):
- Have 3+ conversations across sessions on different topics
- Verify: fragments are extracted, retrieval finds cross-session context, intent shapes responses, profile adapts, insights emerge

End-to-end test after Phase 11 (scale & resilience):
- Re-run the Phase 10 end-to-end test. Outputs should be identical; latency should be materially lower where parallelization applies.
- Inject a synthetic failure mid-pipeline. Verify: the failed unit is logged, the rest of the batch completes, the system exits cleanly rather than crashing.
