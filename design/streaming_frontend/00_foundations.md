# Foundations — Concepts That Appear Everywhere

Read this first. The feature docs reference these ideas without re-explaining them.

---

## 1. LangGraph nodes and state

A LangGraph **graph** is just a directed graph of **nodes**. Each node is a Python
function:

```python
def my_node(state: JournalState) -> dict:
    # read from state, do work, return a partial update
    return {"some_field": new_value}
```

The return value is a **partial dict** — only the fields that changed. LangGraph
merges it back into the full state.

### State reducers

Some fields on `JournalState` are annotated with a **reducer**:

```python
# graph/state.py
transcript: Annotated[list[Exchange], add] = Field(default_factory=list)
session_messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
```

`add` means "append the returned list to the existing list" — not replace it.
`add_messages` does the same but also deduplicates by message ID.

**Why this matters**: when a node returns `{"transcript": [new_exchange]}`, LangGraph
calls `add(existing_transcript, [new_exchange])` to produce the next state. The node
never sees or returns the full list — it just hands back the new items.

Inside a node (or the EOS single-node pipeline), reducers do **not** apply — you are
working with plain Python objects. Only at the graph boundary, when the node's return
dict is merged into state, do reducers fire.

---

## 2. The `make_*` factory pattern

Nodes are built by factory functions rather than defined at module level:

```python
def make_intent_classifier(llm: LLMClient, ...) -> Callable:
    @node_trace("intent_classifier")
    def intent_classifier(state: JournalState) -> dict:
        ...
    return intent_classifier
```

**Why**: the node function needs access to `llm`, `context_builder`, and other
dependencies. Closing over them in a factory avoids passing them through state or using
globals. The graph builder calls `make_intent_classifier(registry.get("classifier"))` at
startup and stores the resulting function as a node.

---

## 3. `astream_events(version="v2")`

Instead of `graph.invoke()` (which returns only the final state), we call:

```python
events = graph.astream_events(input, config=config, version="v2")
async for event in events:
    print(event["event"], event.get("metadata", {}).get("langgraph_node"))
```

Every time something interesting happens inside the graph, it emits an event dict.
The events we care about:

| event name | when it fires | what's in `data` |
|---|---|---|
| `on_chat_model_stream` | each token from the LLM | `chunk` — an `AIMessageChunk` |
| `on_chat_model_end` | LLM call finished | `output` — full `AIMessage` with `usage_metadata` |
| `on_chat_model_error` | LLM call failed | `error` |

Every event has a `metadata` dict that LangGraph populates with `langgraph_node` — the
name of the node that triggered the event. This lets us filter: "only show tokens from
`get_ai_response`, not from `intent_classifier`."

---

## 4. Server-Sent Events (SSE)

SSE is the HTTP streaming protocol the API uses to send tokens to the browser as they
arrive, instead of waiting for the full response.

Wire format — each event is two lines followed by a blank line:

```
event: token
data: {"text": "Hello "}

event: token
data: {"text": "world"}

event: done
data: {"text": ""}
```

The browser's `EventSource` API (or `fetch` with a streaming reader) receives these
chunks as they arrive. FastAPI's `StreamingResponse` handles the HTTP side; our
`graph_stream` generator produces the SSE strings.

Our event types (`api/models.py::SseEvent`):
- `token` — one chunk of AI text
- `system` — feedback from the graph (e.g. `/save` confirmation)
- `done` — stream complete
- `error` — something failed

---

## 5. LangChain callbacks

LangChain's callback system is how it reports internal events (LLM start, LLM end,
chain start, etc.) to external observers. `astream_events` is essentially a callback
consumer exposed as an async generator — same underlying mechanism.

You can also attach a `BaseCallbackHandler` directly to any invocation via
`config={"callbacks": [handler]}`. The handler's methods (`on_llm_end`,
`on_llm_error`, etc.) fire synchronously during execution. LangGraph automatically
passes `langgraph_node` through to the handler's `**kwargs["metadata"]`.

---

## 6. `JournalState` field reference (quick lookup)

| field | reducer | who reads it | who writes it |
|---|---|---|---|
| `session_id` | none | everywhere | set once at session start |
| `session_messages` | `add_messages` | intent_classifier, profile_scanner, get_ai_response | each turn via `build_turn_input` |
| `transcript` | `add` | EOS pipeline | `session_store.on_ai_turn` → graph node |
| `threads` | `add` | thread_classifier | exchange_decomposer |
| `classified_threads` | `add` | fragment_extractor | thread_classifier |
| `context_specification` | none | get_ai_response | intent_classifier |
| `user_profile` | none | profile_scanner, conversation prompt | profile_scanner |
| `system_message` | none | API (after stream) | CAPTURE node (for /save etc.) |
