# #7-hardening — The Classifier Eval Harness

## What changed

A complete eval harness was built in `journal_agent/evals/`. It can:
1. Run all four classifiers against a fixture set and record every output to JSONL.
2. Compare two run files and report what changed — output, prompt version, or both.
3. Be invoked from the command line.

---

## Why this is the right call

A prompt is not code — you can't write a unit test that proves it produces the right
output. The output depends on the model, the input, and the prompt text all at once.
What you *can* do is:

- Run the same input through two versions of a prompt and compare outputs.
- Detect regressions: "this input used to produce 3 threads; now it produces 1."
- Trace every output to the exact prompt version that produced it.

This is an **eval harness**, not a test suite. Tests verify correctness against a known
answer. Evals detect *change* — they don't know if the new output is better, only that
it's different. A human still makes that judgment.

---

## How it works

### Package layout

```
journal_agent/evals/
    fixtures.py    — load fixtures, build JournalState inputs
    runner.py      — run classifiers, produce EvalRecord list, write/load JSONL
    compare.py     — diff two run files, produce human-readable report
    data/
        eval_fixtures.jsonl   — 3 built-in fixture sessions
```

### Fixtures (`evals/fixtures.py`)

A **fixture** is a small, deterministic session: a list of `Exchange` objects that
represent a real conversation. Three built-in fixtures cover the three intent archetypes
the classifiers need to handle:

| fixture_id | description | dominant intent |
|---|---|---|
| `f001` | Personal journaling about work stress | high `first_person_score` |
| `f002` | Factual questions about computer hardware | high `question_score` |
| `f003` | Task requests (write email, summarize) | high `task_score` |

The fixture file (`evals/data/eval_fixtures.jsonl`) is JSONL — one fixture per line,
each deserialised into `Exchange` objects using the same Pydantic models the real app
uses. There is no special fixture format; fixtures are just real app data.

Two state-building functions convert a fixture into the right JournalState shape for
each classifier:

```python
def build_intent_state(fixture: Fixture) -> JournalState:
    # intent_classifier reads state.session_messages
    # → flatten exchanges into HumanMessage / AIMessage list
    messages = []
    for ex in fixture.exchanges:
        if ex.human: messages.append(HumanMessage(content=ex.human.content))
        if ex.ai:    messages.append(AIMessage(content=ex.ai.content))
    return JournalState(session_id=fixture.fixture_id, session_messages=messages)

def build_eos_state(fixture: Fixture) -> JournalState:
    # EOS classifiers read state.transcript
    return JournalState(session_id=fixture.fixture_id, transcript=fixture.exchanges)
```

### The EvalRecord (`evals/runner.py`)

Every classifier call produces one `EvalRecord`:

```python
class EvalRecord(BaseModel):
    fixture_id:     str             # which fixture was used
    classifier:     str             # "intent_classifier", "exchange_decomposer", etc.
    prompt_key:     str             # the PromptKey enum value
    prompt_version: str             # e.g. "v1" — from get_prompt_version()
    input_hash:     str             # 8-char SHA-256 of the serialised input
    output:         dict            # model_dump() of the structured output
    elapsed_ms:     int
    timestamp:      str             # ISO 8601
    error:          str | None      # set if the classifier returned STATUS=ERROR
```

`input_hash` is the key field for comparison. If the same fixture produces the same
input hash across two runs, you know the input didn't change — only the prompt or model
did. If the input hash changes, a fixture was edited.

### The runner (`evals/runner.py::run_suite`)

The EOS classifiers are **chained** — each step's output feeds the next:

```python
# 1. exchange_decomposer (sync)
result = decomposer(eos_state)
threads = result.get("threads", [])

# 2. thread_classifier (async) — receives the threads from step 1
eos_state = eos_state.model_copy(update={"threads": threads})
result = await classifier(eos_state)
classified = result.get("classified_threads", [])

# 3. thread_fragment_extractor (async) — receives classified threads from step 2
eos_state = eos_state.model_copy(update={"classified_threads": classified})
result = await extractor(eos_state)
```

This mirrors the real EOS pipeline. If the decomposer produces 2 threads, the
classifier sees 2 threads. The fragment extractor sees whatever the classifier produced.

Errors are captured without stopping the suite — if `exchange_decomposer` fails, the
`EvalRecord` gets `error="..."` and the classifier and extractor steps still run (with
empty threads as input).

The classifiers are called via their `make_*` factories directly — no LangGraph graph,
no checkpointer, no HTTP. This is exactly what the evaluability work (#7-design) made
possible: because classifiers are plain functions closed over an LLM client, they can
be called in any context.

### The comparator (`evals/compare.py::compare_runs`)

Two JSONL run files are loaded and indexed by `(fixture_id, classifier)`. Records are
joined on that key:

```python
records_a = {(r.fixture_id, r.classifier): r for r in load_results(path_a)}
records_b = {(r.fixture_id, r.classifier): r for r in load_results(path_b)}
```

For each pair, it checks:
- `prompt_version` — did the prompt change?
- `input_hash` — did the fixture content change?
- `output` — did the model produce a different result?

Output diff is field-by-field:

```
  CHANGED exchange_decomposer / f001
    prompt_version: 'v1' → 'v2'
    output changed:
      ~ threads
        A: [{"thread_name": "work_stress", ...}]
        B: [{"thread_name": "work_anxiety", ...}, {"thread_name": "job_search", ...}]
```

---

## The workflow

```bash
# 1. Establish a baseline
uv run python -m journal_agent.scripts.run_evals --output evals/baseline.jsonl

# 2. Edit a prompt — bump its VERSION constant
#    e.g. in configure/prompts/decomposer.py: VERSION = "v2"

# 3. Run again
uv run python -m journal_agent.scripts.run_evals --output evals/after_prompt_change.jsonl

# 4. See what changed
uv run python -m journal_agent.scripts.run_evals \
    --compare evals/baseline.jsonl evals/after_prompt_change.jsonl
```

The comparator output shows every record that changed, with the prompt version bump
and a field-by-field diff of the output. You decide if the change is an improvement.

---

## Before → After

### Before

Evaluating a prompt change meant: change the text, run the whole app, talk to it
manually, judge by feel. No record of what it said before. No way to know if other
classifiers were affected.

### After

Change the prompt, bump VERSION, run `run_evals`, run `compare`. The diff tells you
exactly which fixtures changed, what the output was before, and what it is now.
The record file is a permanent artifact — commit it alongside the prompt change.

---

## Key things to remember

- Fixtures are real `Exchange` objects serialised to JSONL. You can add your own by
  copying real session data into `eval_fixtures.jsonl` and giving it a `fixture_id`.
- `input_hash` is your signal for "same input, different output." If input_hash is
  stable across runs and output changed, something in the prompt or model changed.
- The harness makes real LLM calls — it costs money and takes time. Run it before
  committing a prompt change, not as part of CI.
- Add fixtures whenever you find a case the current prompts handle badly. Over time the
  fixture set becomes a regression suite for prompt quality.
