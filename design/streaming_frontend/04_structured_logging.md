# #8-design — Structured Observability

## What changed

Three improvements to logging, all using Python's stdlib `logging` only:

1. **`node_trace` is always on** — the `NODE_TRACE_ENABLED` env-var gate was removed.
   The decorator now wraps every node unconditionally.
2. **Structured log fields** — log calls switched from format strings to `extra={}`
   dicts. Every record from `node_trace` carries `node`, `session_id`, `elapsed_ms`,
   and `status` as first-class fields, not buried in the message string.
3. **Decision-point logs** — after each classifier runs, a log record captures what it
   decided: scores, counts, routing outcome. API endpoints log session lifecycle events
   the same way.

---

## Why this is the right call

There are two kinds of logging: logging for humans reading a terminal, and logging for
machines aggregating a stream of records.

Format strings produce human-readable messages:
```
"Node intent_classifier completed in 0.423s (session_id=abc, status=ok)"
```

A log aggregator (Datadog, CloudWatch, Splunk, even `jq`) has to *parse* that string
to extract `elapsed_ms` or `session_id`. Parsing is fragile — one character change in
the format string breaks every dashboard query that depends on it.

`extra={}` fields produce structured records:
```json
{"message": "node completed", "node": "intent_classifier", "session_id": "abc",
 "elapsed_ms": 423, "status": "None"}
```

Now a dashboard can filter `node == "intent_classifier" AND elapsed_ms > 2000` without
any string parsing. The message itself becomes a stable human-readable label; the data
travels as structured fields.

---

## How it works

### `node_trace` — always-on wrapper (`graph/node_tracer.py`)

Previously:
```python
_ENABLED = os.getenv("NODE_TRACE_ENABLED", "false").lower() in ("1", "true", "yes")

def node_trace(node_name=None):
    def decorator(func):
        if not _ENABLED:
            return func    # ← gate: function returned unchanged
        ...
```

Now:
```python
def node_trace(node_name=None):
    def decorator(func):
        name = node_name or func.__name__
        # always wrap — no gate
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(state): ...
            return async_wrapper
        @wraps(func)
        def wrapper(state): ...
        return wrapper
```

The wrapper measures `perf_counter()` before and after the call, then logs:

```python
logger.info(
    "node completed",
    extra={
        "node": name,
        "session_id": session_id,
        "elapsed_ms": round(elapsed * 1000),
        "status": str(status),
    },
)
```

If the node returns `status=ERROR`, it logs at `WARNING` level and includes
`error_message`. If the node raises an exception (unexpected), it logs at `ERROR` with
`exc_info=True` so the full traceback appears.

### Decision-point logs in classifiers (`graph/nodes/classifiers.py`)

After each classifier produces its output, a log record captures the decision:

```python
# intent_classifier — after score_card is computed and specification resolved
logger.info(
    "intent_classifier decision",
    extra={
        "session_id": state.session_id,
        "question_score": score_card.question_score,
        "first_person_score": score_card.first_person_score,
        "personalization_score": score_card.personalization_score,
        "task_score": score_card.task_score,
        "prompt_key": specification.prompt_key.value,  # which response strategy was chosen
    },
)

# exchange_decomposer — after thread list is returned
logger.info("exchange_decomposer decision",
    extra={"session_id": state.session_id, "thread_count": len(thread_list.threads)})

# thread_classifier — after fan-out gather
logger.info("thread_classifier decision",
    extra={"session_id": state.session_id, "thread_count": len(classified_threads)})

# thread_fragment_extractor — after flattening
logger.info("thread_fragment_extractor decision",
    extra={"session_id": state.session_id, "fragment_count": len(fragments)})
```

These records let you ask: "for session X, what did the intent classifier decide, and
why?" Or: "how many fragments did the EOS pipeline extract across all sessions?"

### API boundary logs (`api/main.py`)

```python
# POST /sessions
logger.info("session created", extra={"session_id": session_id})

# POST /chat/{session_id}
logger.info("turn received", extra={"session_id": session_id, "message_len": len(request.message)})

# DELETE /sessions/{session_id}
logger.info("eos triggered", extra={"session_id": session_id})
logger.info("eos completed", extra={"session_id": session_id})
logger.error("eos pipeline failed", extra={"session_id": session_id, "error": str(exc)})
```

These bracket the session lifecycle. By filtering on `session_id`, you can reconstruct
the complete timeline of a session from its logs.

---

## A note on `config_builder.py`

`configure_environment()` sets `propagate=False` on the `node_tracer` logger and
routes it to a file (`journal_agent.log`). This prevents node trace output from
cluttering the terminal during interactive sessions.

The observability tests need `caplog` to capture these records, but `propagate=False`
prevents that. The test file adds an `autouse` fixture that temporarily re-enables
propagation for the duration of each test, then restores it:

```python
@pytest.fixture(autouse=True)
def restore_node_tracer_propagation():
    nt = logging.getLogger("journal_agent.graph.node_tracer")
    orig = nt.propagate
    nt.propagate = True
    yield
    nt.propagate = orig
```

This is a pattern worth knowing: when testing code that has been deliberately configured
to suppress output, restore the configuration temporarily rather than fighting the
suppression.

---

## Before → After

### Before

```python
# node_tracer.py — gated, format strings
_ENABLED = os.getenv("NODE_TRACE_ENABLED", "false")  # off by default

logger.info(
    "Node %s completed in %.3fs (session_id=%s, status=%s)",
    name, elapsed, session_id, status
)
# → "Node intent_classifier completed in 0.423s (session_id=abc, status=None)"
# A log aggregator has to parse this string to get elapsed or session_id.
```

### After

```python
# node_tracer.py — always on, structured
logger.info(
    "node completed",
    extra={"node": name, "session_id": session_id, "elapsed_ms": 423, "status": "None"}
)
# → LogRecord with .node, .session_id, .elapsed_ms, .status as direct attributes
# A log aggregator can filter/group on these without any parsing.
```

---

## Key things to remember

- `extra={}` fields become attributes on the `LogRecord` object. Any log formatter or
  handler that formats JSON (like `python-json-logger`) will include them automatically.
- The message string should be a stable, human-readable *label*. The data goes in
  `extra`. This way you can change the message without breaking queries.
- Decision-point logs go *inside* the classifier, after the LLM call returns — they
  capture what the model actually decided, not what we asked it.
- `node_trace` wraps at decoration time (when `make_*` is called at startup). The
  timing measurement happens at call time (each turn).
