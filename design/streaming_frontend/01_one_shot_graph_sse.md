# #5 / #9c — One-Shot Graph Invocation and the FastAPI SSE Stream

## What changed

The conversation graph stopped looping. The API stopped faking a stream.

**Before**: the graph had a `get_user_input` node that called `input()` in a loop,
keeping the graph alive across turns. The terminal ran it once and the graph cycled
internally. The FastAPI endpoint generated fake tokens from a hard-coded string.

**After**: the graph runs exactly once per turn — starts at `intent_classifier`, ends
after `get_ai_response`. The terminal client loops in Python. The API invokes the graph
once per HTTP request and streams real tokens back over SSE.

---

## Why this is the right call

The `get_user_input` loop inside the graph was an awkward fit for HTTP. HTTP is
request/response: a client sends a message, the server replies, the connection closes
(or in streaming, closes after the response finishes). There is no way to keep an HTTP
connection open so the graph can `input()` again.

The fix isn't a workaround — it's the correct architecture. Each HTTP request is one
turn. The graph handles that turn and exits. The checkpointer (Postgres) preserves state
between turns so the graph picks up where it left off next time.

A secondary benefit: the `recursion_limit=1000` band-aid in `main.py` could be removed.
That limit existed only to stop an infinitely-looping graph from running forever. Once
the graph is one-shot, there is no loop to limit.

---

## How it works

### The graph topology (`graph/journal_graph.py`)

The conversation graph now has a simple linear spine:

```
intent_classifier → profile_scanner → get_ai_response
```

No `get_user_input` node. The graph compiles once at startup and is invoked once per
turn with `astream_events`.

### Building the turn input (`comms/commands.py::build_turn_input`)

Each HTTP request body becomes a dict that seeds the graph's input for this turn:

```python
turn_input = build_turn_input(parsed, session_id=session_id)
# → {"session_messages": [HumanMessage(content="...")], "session_id": "..."}
```

The graph receives that dict, merges it into checkpointed state, and runs.

### The streaming endpoint (`api/main.py::chat`)

```python
config = {"configurable": {"thread_id": session_id}, "callbacks": [...]}
events = app.state.conversation.astream_events(turn_input, config=config, version="v2")
return StreamingResponse(graph_stream(events, ...), media_type="text/event-stream")
```

`astream_events` returns an async generator of event dicts. `graph_stream` filters them
and formats the SSE strings. `StreamingResponse` holds the HTTP connection open until
the generator is exhausted.

### The SSE generator (`api/streaming.py::graph_stream`)

```python
async for event in events:
    if event["event"] == "on_chat_model_stream":
        if event["metadata"]["langgraph_node"] == "get_ai_response":
            chunk = event["data"]["chunk"]
            if chunk.content:
                yield format_sse(SseEvent.TOKEN, chunk.content)

# after stream finishes:
snapshot = await conversation.aget_state(config)
if msg := snapshot.values.get("system_message"):
    yield format_sse(SseEvent.SYSTEM, msg)
yield format_sse(SseEvent.DONE, "")
```

Two things worth noting:
1. We filter by `langgraph_node == "get_ai_response"` — without this, tokens from the
   `intent_classifier` LLM call would leak into the user-facing stream.
2. `system_message` is read from the checkpointer *after* the stream ends. The CAPTURE
   node (handling `/save` etc.) writes it to state; we read it back here. This is why
   the `system` SSE event always comes last before `done`.

### First-turn bootstrap (`api/main.py`)

The first turn for a session needs `user_profile` and `recent_messages` (context from
previous sessions). The API tracks which sessions are new:

```python
if session_id in app.state.new_sessions:
    turn_input["user_profile"] = app.state.user_profile
    turn_input["recent_messages"] = app.state.seed_context
    app.state.new_sessions.discard(session_id)
```

After the first turn the checkpointer holds the full state, so subsequent turns don't
need bootstrapping.

---

## Before → After

### Before

```
Terminal                    Graph
  │                           │
  ├── invoke(graph) ──────────►
  │                           ├── intent_classifier
  │                           ├── get_ai_response  (prints to stdout)
  │                           ├── get_user_input   (blocks on input())
  │                           ├── intent_classifier
  │                           ├── get_ai_response
  │                           ├── ...              (loops forever)
  │◄────────────────── raises RecursionError ──────┤
```

API served a fake stream — `"This is a fake response"` split into tokens with asyncio
sleeps. No graph was involved.

### After

```
HTTP client          FastAPI              LangGraph graph
     │                  │                      │
     ├── POST /chat ────►                      │
     │                  ├── astream_events() ──►
     │                  │                      ├── intent_classifier
     │◄── token ────────┤◄── on_chat_model_stream (intent tokens, filtered out)
     │                  │                      ├── get_ai_response
     │◄── token ────────┤◄── on_chat_model_stream (ai tokens, passed through)
     │◄── token ────────┤◄── ...
     │                  │                      └── (graph exits)
     │◄── system ───────┤◄── read checkpointed system_message
     │◄── done ─────────┤
     │                  │
```

One HTTP request. One graph invocation. Real tokens. Clean exit.

---

## Key things to remember

- The graph is stateless between requests — state lives in the Postgres checkpointer,
  keyed by `thread_id = session_id`.
- `astream_events` replaces `invoke`: same graph, same nodes, just exposes internal
  events as an async stream instead of waiting for the final return value.
- The `langgraph_node` filter in `graph_stream` is critical — every LLM call in the
  graph (including classifiers) emits `on_chat_model_stream` events.
