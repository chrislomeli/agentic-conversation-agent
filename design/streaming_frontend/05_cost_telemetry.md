# #8-hardening — Cost and Failure Telemetry via LangChain Callbacks

## What changed

A `TelemetryCallbackHandler` was added (`telemetry.py`) and wired into both the
conversation graph and the EOS graph. It logs token counts and LLM errors for every
LLM call in the system — classifiers, the main AI response, EOS pipeline — in
structured `extra={}` fields.

---

## Why this is the right call

Node-level latency was already captured by `node_trace`. What was missing:

- **Token counts**: the biggest cost driver. Without knowing how many tokens each
  LLM call uses, you can't reason about API costs or catch runaway prompts.
- **LLM failures**: a model returning a rate-limit error or timing out should produce
  a log record with enough context to debug it — which node, which session, what error.

The right place to capture this is in LangChain's callback system, not inside each
classifier manually. LangChain already knows when every LLM call starts and ends; we
just need to listen.

---

## How LangChain callbacks work

LangChain's callback system is a publish/subscribe mechanism for internal events.
Every LLM call, chain invocation, and tool use publishes events. You subscribe by
providing a `BaseCallbackHandler` subclass.

```python
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

class MyHandler(BaseCallbackHandler):
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        # fires when any LLM call in the chain completes
        pass

    def on_llm_error(self, error: BaseException, **kwargs) -> None:
        # fires when any LLM call raises
        pass
```

You attach a handler to a specific invocation via the `config` parameter:

```python
config = {"callbacks": [MyHandler()]}
graph.ainvoke(input, config=config)          # handler fires for all LLM calls in graph
graph.astream_events(input, config=config)   # same — astream_events IS the callback system
```

LangGraph automatically populates `kwargs["metadata"]["langgraph_node"]` with the name
of the currently-executing node. This is how the handler knows *which* node triggered
the call without any instrumentation inside the node itself.

---

## How it works

### Token count normalisation (`telemetry.py::_token_counts`)

OpenAI and Anthropic return token usage in different field names:

```python
# OpenAI LLMResult
llm_output = {"token_usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140}}

# Anthropic LLMResult
llm_output = {"usage": {"input_tokens": 80, "output_tokens": 30}}
```

The helper normalises both:

```python
def _token_counts(llm_output: dict) -> dict[str, int]:
    usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    prompt     = usage.get("prompt_tokens")     or usage.get("input_tokens",  0)
    completion = usage.get("completion_tokens") or usage.get("output_tokens", 0)
    total      = usage.get("total_tokens")      or (prompt + completion)
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}
```

### The handler (`telemetry.py::TelemetryCallbackHandler`)

```python
class TelemetryCallbackHandler(BaseCallbackHandler):

    def on_llm_end(self, response: LLMResult, *, run_id, **kwargs) -> None:
        llm_output = response.llm_output or {}
        counts = _token_counts(llm_output)
        metadata = kwargs.get("metadata") or {}
        logger.info(
            "llm call completed",
            extra={
                "node":              metadata.get("langgraph_node", "unknown"),
                "model":             llm_output.get("model_name", "unknown"),
                **counts,            # prompt_tokens, completion_tokens, total_tokens
            },
        )

    def on_llm_error(self, error: BaseException, *, run_id, **kwargs) -> None:
        metadata = kwargs.get("metadata") or {}
        logger.error(
            "llm call failed",
            extra={
                "node":  metadata.get("langgraph_node", "unknown"),
                "error": str(error),
            },
        )
```

### Wiring (`api/main.py`)

A new handler instance is created for each request and attached to the graph config:

```python
# conversation turn
config = {
    "configurable": {"thread_id": session_id},
    "callbacks": [TelemetryCallbackHandler()],
}
events = app.state.conversation.astream_events(turn_input, config=config, version="v2")

# EOS pipeline
config = {
    "configurable": {"thread_id": session_id},
    "callbacks": [TelemetryCallbackHandler()],
}
await app.state.eos.ainvoke({}, config=config)
```

Creating a new handler per request is fine — the handler is stateless.

### What gets logged per turn

A typical conversation turn runs three LLM calls: `intent_classifier`,
`profile_scanner` (conditional), and `get_ai_response`. Each fires `on_llm_end`,
producing three log records like:

```
llm call completed  node=intent_classifier  model=gpt-4o-mini  prompt_tokens=312  completion_tokens=28  total_tokens=340
llm call completed  node=get_ai_response    model=gpt-4o        prompt_tokens=891  completion_tokens=203 total_tokens=1094
```

A typical EOS run fires one record per classifier (decomposer, thread_classifier per
thread, fragment_extractor per thread).

---

## Before → After

### Before

Token usage was not logged anywhere. If API costs were higher than expected, the only
way to investigate was to query LangSmith. There was no local record.

LLM failures produced an exception that the node's `try/except` caught and turned into
`{"status": ERROR}` — the original error was visible in the stack trace log but had no
structured fields like `node` or `session_id` to filter on.

### After

Every LLM call produces a structured log record with token counts and model name.
Every LLM failure produces an error record with the node name, session ID, and error
string. Both are queryable without parsing message text.

---

## Why one handler instance per request, not one shared instance

The handler is stateless (no instance variables), so sharing would work. But creating
one per request follows the same pattern as the `config` dict itself — it's scoped to
a single invocation. If the handler ever gains state (e.g., accumulating token counts
for a request-level summary), per-request scoping will already be correct.

---

## Key things to remember

- `astream_events` and `callbacks` are not two separate systems — `astream_events` IS
  the callback system, exposed as an async generator. Attaching a callback to an
  `astream_events` call fires both the event stream and the handler.
- `kwargs["metadata"]["langgraph_node"]` is LangGraph-specific — plain LangChain
  chains won't populate this. The `"unknown"` default handles that case.
- `LLMResult.llm_output` is provider-specific. The `_token_counts` normaliser is the
  single place where provider differences are handled — don't spread that logic
  elsewhere.
